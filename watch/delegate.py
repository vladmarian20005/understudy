"""Where an object goes after a hand-to-hand pass in a score (watch/SCORE.md)."""
# when the score can't say (a moving head camera): in front of the taker
DEFAULT_TARGET = {"left": [-12.0, 24.0], "right": [12.0, 24.0]}


def target_for(score, override=None):
    """Where the object should end: override, else the score's own (overhead clips), else in front of the taker."""
    if override is not None:
        return [float(v) for v in override]
    if score.get("target_cm") is not None:
        return [float(v) for v in score["target_cm"]]
    return list(DEFAULT_TARGET[score["taker"]])
