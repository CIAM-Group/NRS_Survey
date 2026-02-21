from typing import Callable, TypeVar
from inspect import signature


T = TypeVar('T')


def with_invalid_kwargs_filtered(fn: Callable[..., T]) -> Callable[..., T]:
    valid_keywords = list(signature(fn).parameters.keys())
    def wrapped_fn(*args, **kwargs):
        return fn(*args, **{key: value for key, value in kwargs.items() if key in valid_keywords})
    return wrapped_fn
