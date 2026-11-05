"""One owned input snapshot shared by a public calculation's native stages."""

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

_OWNER = ContextVar("behave_fire_native_input_owner", default=None)


class _InputOwner:
    """Retain snapshots and their identities until the public call returns."""

    def __init__(self):
        """Create a memo retaining both source objects and owned results.

        :return: None; initializes this calculation's identity memo.
        """
        self.memo = {}

    def remember(self, value):
        """Register fresh results so later stages can reuse them and their views.

        :param value: Owned native result or nested result container.
        :return: The unchanged result, retained until the calculation ends.
        """
        if isinstance(value, (np.ndarray, dict, tuple, list)):
            self.memo[id(value)] = (value, value)
            if isinstance(value, dict):
                for item in value.values():
                    self.remember(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    self.remember(item)
            elif np.ma.isMaskedArray(value):
                self.remember(value._data)
                self.remember(value._mask)
        return value

    def snapshot(self, value):
        """Copy caller buffers once while retaining scalar and object payload kinds.

        :param value: Input value, container, or view used by a native stage.
        :return: Owned equivalent, reusing existing snapshots and owned views.
        """
        previous = self.memo.get(id(value))
        if previous is not None:
            return previous[1]
        if np.ma.isMaskedArray(value):
            result = np.ma.array(self.snapshot(value._data), mask=self.snapshot(value._mask), copy=False)
            if value._fill_value is not None:
                result._fill_value = self.snapshot(value._fill_value)
        elif isinstance(value, np.ndarray):
            base = value
            while isinstance(base, np.ndarray):
                previous = self.memo.get(id(base))
                if previous is not None and previous[0] is previous[1]:
                    return self.remember(value)
                base = base.base
            result = value.copy(order="K")
        elif isinstance(value, dict):
            result = {}
            self.memo[id(value)] = (value, result)
            result.update((key, self.snapshot(item)) for key, item in value.items())
        elif isinstance(value, tuple):
            result = tuple(self.snapshot(item) for item in value)
        elif isinstance(value, list):
            result = []
            self.memo[id(value)] = (value, result)
            result.extend(self.snapshot(item) for item in value)
        else:
            return value
        self.memo[id(value)] = (value, result)
        return self.remember(result)


@contextmanager
def _owned_inputs():
    """Share ownership across nested stages and restore context after exceptions.

    :return: Scoped input owner, released when the outer public call returns.
    """
    owner = _OWNER.get()
    token = None
    if owner is None:
        owner = _InputOwner()
        token = _OWNER.set(owner)
    try:
        yield owner
    finally:
        if token is not None:
            _OWNER.reset(token)
