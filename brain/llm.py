"""The one LLM seam: chat(system, messages, tools) -> Reply, the same for every model the brain can use.

Messages and tools are in the OpenAI chat format (images as image_url data URLs). LLM= in the environment or .env
picks the backend:
    muse      Muse Spark through Meta's Model API (OpenAI SDK). MODEL_API_KEY, MUSE_MODEL (default muse-spark-1.3),
              MODEL_API_BASE (default https://api.meta.ai/v1)
    <name>    brain/llm_<name>.py, a local backend that stays out of git (its module has a Backend class)
Tests pass a Scripted backend in directly. Every number we report comes from Muse Spark.
"""
import importlib
import json
import os
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
PRICES = {"muse-spark": (1.25, 4.25)}  # $ per million input, output tokens


def load_env(path=os.path.join(HERE, "..", ".env")):
    """KEY=value lines from .env into the environment, without overriding what's already set."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    error: str = None  # the arguments weren't valid JSON


@dataclass
class Reply:
    text: str
    calls: list
    message: dict  # the assistant message to append to the history
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    model: str = ""

    @property
    def cost(self):
        price = next((p for k, p in PRICES.items() if self.model.startswith(k)), (0.0, 0.0))
        return (self.tokens_in * price[0] + self.tokens_out * price[1]) / 1e6


def parse_calls(tool_calls):
    out = []
    for tc in tool_calls or []:
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
            out.append(ToolCall(tc["id"], tc["function"]["name"], args if isinstance(args, dict) else {}))
        except json.JSONDecodeError as e:
            out.append(ToolCall(tc["id"], tc["function"]["name"], {}, f"arguments aren't valid JSON: {e}"))
    return out


class Muse:
    def __init__(self):
        from openai import OpenAI
        key = os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY")
        if not key:
            raise SystemExit("no MODEL_API_KEY: put it in .env (MODEL_API_KEY=...)")
        self.model = os.environ.get("MUSE_MODEL", "muse-spark-1.3")
        self.client = OpenAI(base_url=os.environ.get("MODEL_API_BASE", "https://api.meta.ai/v1"), api_key=key,
                             timeout=120, max_retries=3)

    def chat(self, system, messages, tools):
        t0 = time.time()
        r = self.client.chat.completions.create(model=self.model, messages=[dict(role="system", content=system)]
                                                + messages, tools=tools)
        m = r.choices[0].message
        calls = [dict(id=tc.id, type="function", function=dict(name=tc.function.name,
                                                                arguments=tc.function.arguments))
                 for tc in m.tool_calls or []]
        message = dict(role="assistant", content=m.content or "")
        if calls:
            message["tool_calls"] = calls
        u = r.usage
        return Reply(m.content or "", parse_calls(calls), message, u.prompt_tokens if u else 0,
                     u.completion_tokens if u else 0, time.time() - t0, self.model)

    def ask(self, system, messages, effort=None):
        """A plain answer, no tools (the planner's JSON). effort: the model's reasoning effort ("high" to think a
        plan through, "low" for quick checks); None: the model's default."""
        t0 = time.time()
        extra = {} if effort is None else {"reasoning_effort": effort}
        r = self.client.chat.completions.create(model=self.model, messages=[dict(role="system", content=system)]
                                                + messages, **extra)
        text = r.choices[0].message.content or ""
        u = r.usage
        return Reply(text, [], dict(role="assistant", content=text), u.prompt_tokens if u else 0,
                     u.completion_tokens if u else 0, time.time() - t0, self.model)


class Scripted:
    """Plays back a list of replies, each text or (name, args) tool calls: for tests, no model involved."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []  # the messages each call was given
        self.model = "scripted"

    def chat(self, system, messages, tools):
        self.seen.append(list(messages))
        step = self.steps.pop(0) if self.steps else "(script over)"
        if isinstance(step, str):
            return Reply(step, [], dict(role="assistant", content=step), model=self.model)
        calls = [dict(id=f"call_{len(self.seen)}_{k}", type="function",
                      function=dict(name=name, arguments=json.dumps(args))) for k, (name, args) in enumerate(step)]
        return Reply("", parse_calls(calls), dict(role="assistant", content="", tool_calls=calls), model=self.model)


def backend(name=None):
    load_env()
    name = name or os.environ.get("LLM", "muse")
    if name == "muse":
        return Muse()
    return importlib.import_module(f"brain.llm_{name}").Backend()
