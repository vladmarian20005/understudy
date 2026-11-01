"""The brain's model, narrated: every reply is reported the moment it arrives, so the page can show Muse thinking while
it drives the arms.

What Muse says before each call is in the reply's text (the brain's prompt asks it to say what it sees and what it'll
do). When the API also returns the model's reasoning (a reasoning_content or reasoning field on the message), that is
caught too and shown above the text.
"""


def reasoning_of(response):
    """The reasoning text of a chat completion, if the API sent one."""
    try:
        m = response.choices[0].message
    except (AttributeError, IndexError):
        return None
    extra = getattr(m, "model_extra", None) or {}
    for value in (getattr(m, "reasoning_content", None), extra.get("reasoning_content"), extra.get("reasoning")):
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):  # {"content": ...} or {"summary": ...}
            text = value.get("content") or value.get("summary") or value.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def with_reasoning(backend):
    """Catch the reasoning of every reply the brain's Muse backend gets, as backend.last_reasoning. Other backends
    are returned as they are."""
    client = getattr(backend, "client", None)
    if client is None:
        return backend
    completions = client.chat.completions
    create = completions.create

    def create_and_keep(*args, **kwargs):
        response = create(*args, **kwargs)
        backend.last_reasoning = reasoning_of(response)
        return response

    completions.create = create_and_keep
    backend.last_reasoning = None
    return backend


class Narrated:
    """A backend that reports each reply: on_reply(turn, reply, reasoning)."""

    def __init__(self, inner, on_reply):
        self.inner = inner
        self.on_reply = on_reply
        self.turn = 0

    @property
    def model(self):
        return getattr(self.inner, "model", "")

    def chat(self, system, messages, tools):
        if hasattr(self.inner, "last_reasoning"):
            self.inner.last_reasoning = None
        reply = self.inner.chat(system, messages, tools)
        self.turn += 1
        self.on_reply(self.turn, reply, getattr(self.inner, "last_reasoning", None))
        return reply
