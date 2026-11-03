"""Muse's reasoning, for the page: what Muse says is in each reply's text (what it sees, its plan, its action), and when
the API also returns the model's reasoning (a reasoning_content or reasoning field on the message), that is caught too
and shown above the text.
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
