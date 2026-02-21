from typing import Collection, Any


def maybe_eval(s: str | Any, should_keep: Collection = []):
    if not isinstance(s, str):
        return s
    if s in should_keep:
        return s
    return eval(s)
