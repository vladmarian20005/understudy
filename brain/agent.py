"""The agent loop: a task, the overhead image, and tool calls until the model calls done.

Each turn the model sees the history and the newest images (older images are dropped from the history to keep the
turns cheap; older results lose their positions), calls one tool or several in a row (run in order, the rest
skipped after one that doesn't finish), and gets the results plus fresh images: the overhead camera and the wrist
camera of each arm that moved. It ends when the model calls done, after max_turns, when a person stops the arms (the
stop key, or taking an arm over with its leader), or when robotd stops answering. With run_dir, every image the model
is sent is saved there as <turn>_<camera>.jpg and every turn is appended to turns.jsonl as it happens (what the model
said, the call, the whole result), so a run can be checked against what it saw, even one that was killed.
"""
import base64
import json
import os
import time

from harness import robot

from . import tools
from .prompt import SYSTEM

KEEP_IMAGES = 2  # the latest looks the model still sees as images
KEEP_RESULTS = 3  # the latest tool results the model still sees in full
MAX_NUDGES = 3  # replies in a row without a tool call before giving up


def image_part(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def seen_message(text, seen):
    parts = [{"type": "text", "text": text}]
    for cam, b64 in (seen or {}).get("images", {}).items():
        parts += [{"type": "text", "text": f"{cam} camera:"}, image_part(b64)]
    return {"role": "user", "content": parts}


def save_images(run_dir, turn, seen):
    if run_dir is None:
        return
    os.makedirs(run_dir, exist_ok=True)
    for cam, b64 in (seen or {}).get("images", {}).items():
        with open(os.path.join(run_dir, f"{turn:02d}_{cam}.jpg"), "wb") as f:
            f.write(base64.b64decode(b64))


def save_turn(run_dir, turn, **record):
    if run_dir is None:
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "turns.jsonl"), "a") as f:
        f.write(json.dumps(dict(turn=turn, t=round(time.time(), 2), **record), default=str) + "\n")


def shorten_old_results(messages, keep=KEEP_RESULTS):
    """Cut all but the newest keep tool results down to their first line (the outcome), in place: the full positions
    are in the newest ones, and a run's history otherwise grows by a whole state every call (31 Oct: 21k tokens a
    turn by turn 40)."""
    results = [m for m in messages if m.get("role") == "tool" and "\nnow: " in m.get("content", "")]
    for m in results[:-keep] if keep else results:
        m["content"] = m["content"].split("\nnow: ", 1)[0]


def drop_old_images(messages, keep=KEEP_IMAGES):
    """Replace all but the newest keep messages' images with a note, in place."""
    with_images = [m for m in messages if isinstance(m.get("content"), list)
                   and any(p.get("type") == "image_url" for p in m["content"])]
    for m in with_images[:-keep] if keep else with_images:
        m["content"] = [p for p in m["content"] if p.get("type") != "image_url"] + [
            {"type": "text", "text": "(older images dropped)"}]


def run(task, llm, max_turns=40, log=print, run_dir=None):
    """Do one task. Returns dict(success, summary, turns, calls, tokens_in, tokens_out, cost, seconds, transcript)."""
    t0 = time.time()
    tools.reset()
    seen = tools.look()
    save_images(run_dir, 0, seen)
    first = f"Task: {task}\n\nNow: {tools.state_text(seen)}"
    save_turn(run_dir, 0, said=first)
    messages = [seen_message(first, seen)]
    stats = dict(turns=0, calls=[], tokens_in=0, tokens_out=0, cost=0.0, llm_seconds=0.0)
    result, nudges = None, 0
    while result is None and stats["turns"] < max_turns:
        drop_old_images(messages)
        shorten_old_results(messages)
        reply = llm.chat(SYSTEM, messages, tools.TOOLS)
        stats["turns"] += 1
        stats["tokens_in"] += reply.tokens_in
        stats["tokens_out"] += reply.tokens_out
        stats["cost"] += reply.cost
        stats["llm_seconds"] += reply.seconds
        messages.append(reply.message)
        if reply.text.strip():
            log(f"[{stats['turns']}] {reply.text.strip()}")
        save_turn(run_dir, stats["turns"], model=reply.text.strip())
        if not reply.calls:
            nudges += 1
            if nudges >= MAX_NUDGES:
                result = dict(success=False, summary="the model stopped calling tools")
                break
            messages.append({"role": "user", "content": "Call a tool: act, look, or done when it's finished."})
            continue
        nudges = 0
        newest, halted = None, None
        for call in reply.calls:  # in order; after one that didn't finish, the rest are skipped
            if halted and call.name != "done":
                text = f"skipped: {halted} didn't finish"
            elif call.name == "done":
                result = dict(success=bool(call.args.get("success")), summary=str(call.args.get("summary", "")))
                text = "ok"
            elif call.error or call.name not in tools.NAMES:
                text = call.error or f"there's no tool {call.name!r}"
            else:
                log(f"  -> {call.name} {json.dumps(call.args)}")
                try:
                    text, seen, r = tools.run(call.name, call.args)
                    newest = seen or newest
                except robot.RobotdDown as e:
                    text, r = str(e), None
                    result = dict(success=False, summary=f"robotd stopped answering: {e}")
                log(f"     {text.splitlines()[0][:300]}")
                if r and r.get("outcome") not in (None, "done"):
                    halted = call.name
                # a person stopped it: the stop key (no fail kind) or a leader taking an arm over; not a reflex
                if r and r.get("outcome") == "stopped" and r.get("fail") in (None, "manual") and result is None:
                    result = dict(success=False, summary=f"a person stopped the arms: {r.get('why')}")
            save_turn(run_dir, stats["turns"], call=call.name, args=call.args, result=text)
            stats["calls"].append(dict(name=call.name, args=call.args, result=text.splitlines()[0][:500]))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        if newest is not None and result is None:
            save_images(run_dir, stats["turns"], newest)
            messages.append(seen_message("After that:", newest))
    if result is None:
        result = dict(success=False, summary=f"no done after {max_turns} turns")
    log(f"{'success' if result['success'] else 'failure'}: {result['summary']}")
    stats.update(result, cost=round(stats["cost"], 4), llm_seconds=round(stats["llm_seconds"], 1),
                 seconds=round(time.time() - t0, 1), model=getattr(llm, "model", ""), transcript=messages)
    return stats
