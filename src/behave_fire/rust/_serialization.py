"""Versioned value-only snapshots for built-in native objects."""
from importlib import import_module

from .._backend_contract import BINDING_API_VERSION

def _require_extension():
    """Require the compatible native ABI when restoring an explicit native object.

    :return: None; raises an actionable ImportError when native restoration is unavailable.
    """
    try:
        extension = import_module('behave_fire._rust')
    except Exception as error:
        raise ImportError(f'Cannot restore a Rust behave-fire object; install a compatible native wheel or rebuild the extension. {error}') from error
    if getattr(extension, '__binding_api_version__', None) != BINDING_API_VERSION:
        raise ImportError(f'Cannot restore a Rust behave-fire object: binding API version {BINDING_API_VERSION} is required. Rebuild or reinstall the native extension.')

class NativeState:
    """Preserve numerical snapshots without pointers, pools, or backend reselection."""

    def __getstate__(self):
        """Export a versioned envelope while retaining normal shallow-copy semantics.

        :return: Backend/API identifiers and the object's current value/array state.
        """
        return {'schema_version': 1, 'binding_api_version': BINDING_API_VERSION, 'backend': 'rust', 'values': dict(vars(self))}

    def __setstate__(self, state):
        """Restore the recorded native state without rebuilding cached lookup arrays.

        :param state: Versioned value-only snapshot produced by this implementation.
        :return: None; replaces the receiver state after schema and ABI validation.
        """
        if not isinstance(state, dict) or state.get('schema_version') != 1:
            raise ValueError('Unsupported native behave-fire state version')
        if state.get('binding_api_version') != BINDING_API_VERSION or state.get('backend') != 'rust':
            raise ValueError('Incompatible native behave-fire state API or backend')
        values = state.get('values')
        if not isinstance(values, dict):
            raise TypeError('Native behave-fire serialized values must be a dictionary')
        _require_extension()
        vars(self).clear()
        vars(self).update(values)
