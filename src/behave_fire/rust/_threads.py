"""Scoped worker settings for historical static and bound model calls."""
from contextvars import ContextVar
from functools import wraps
from .._ownership import _owned_inputs

_THREADS = ContextVar("behave_fire_native_threads", default=1)


class _ThreadedCall:
    """Bind model calls while retaining historical class-level static calls."""

    def __get__(self, instance, owner):
        """Return a call wrapper with this instance's scoped worker request.

        :param instance: Model instance, or None for a class-level access.
        :param owner: Declaring model class.
        :return: Callable restoring the previous context even after an exception.
        """
        target = self.function if self.static else self.function.__get__(instance, owner)

        @wraps(target)
        def call(*args, **kwargs):
            """Apply and restore the worker request around one model operation.

            :param args: Positional arguments for the original method.
            :param kwargs: Keyword arguments for the original method.
            :return: Original method result.
            """
            model = instance
            if model is None and not self.static:
                model = args[0] if args else kwargs.get("self")
            token = _THREADS.set(1 if model is None else model.threads)
            try:
                with _owned_inputs() as owner:
                    return target(*owner.snapshot(args), **owner.snapshot(kwargs))
            finally:
                _THREADS.reset(token)

        return call

    def __init__(self, function, static):
        """Store the original method and its historical binding convention.

        :param function: Original model method.
        :param static: Whether class access is a historical static call.
        :return: None; saves method metadata.
        """
        self.function = function
        self.static = static


def _threaded_method(function):
    """Wrap an instance method with a scoped native worker request.

    :param function: Original method.
    :return: Descriptor preserving instance binding.
    """
    return _ThreadedCall(function, False)


def _threaded_static(function):
    """Wrap a historical static method while honoring selected instance workers.

    :param function: Original static method.
    :return: Descriptor supporting both class-level and instance-level calls.
    """
    return _ThreadedCall(function, True)
