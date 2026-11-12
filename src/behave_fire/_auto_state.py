"""Value-only FireSize migration for verified complex state operations."""
import copy
from functools import wraps
import importlib
import inspect

from ._auto_routing import _complex_sites

_SUPPORTED = frozenset({'calculate_backing_spread_rate', 'calculate_flanking_spread_rate',
                        'get_eccentricity', 'get_fire_length_to_width_ratio', 'get_heading_to_backing_ratio'})


def _restore_fire_size(backend, values):
    """Restore the recorded delegate without automatic backend reselection.

    :param backend: Previously pinned backend.
    :param values: Complete owned physical state.
    :return: Automatic proxy using the recorded delegate.
    """
    if backend not in ('python', 'rust'):
        raise ValueError('Invalid automatic FireSize backend')
    module = importlib.import_module('behave_fire.' + backend)
    state = AutoFireSize.__new__(AutoFireSize)
    object.__setattr__(state, '_delegate', module.FireSize())
    vars(state._delegate).update(values)
    object.__setattr__(state, '_backend', backend)
    object.__setattr__(state, '_status', {'backend': backend})
    return state


def fire_size_type(status):
    """Bind namespace completion reporting to an independently owned delegate.

    :param status: Shared last-completed backend dictionary.
    :return: FireSize-compatible automatic class.
    """
    return type('FireSize', (AutoFireSize,), {'_namespace_status': status, '__module__': __name__})


class AutoFireSize:
    """Select Rust for real state and pin Python before proven complex operations."""

    _namespace_status = None

    def __getattr__(self, name):
        """Forward fields or preflight a public method before invoking it once.

        :param name: Delegate field or method.
        :return: Value or signature-preserving bound callable.
        """
        target = getattr(self._delegate, name)
        if not callable(target) or name.startswith('_'):
            return target

        @wraps(target)
        def call(*args, **kwargs):
            """Snapshot complete state before any supported backend migration.

            :param args: Original positional inputs.
            :param kwargs: Original named inputs.
            :return: Delegate result; errors propagate without native replay.
            """
            bound = inspect.signature(target).bind(*args, **kwargs)
            sites = _complex_sites(vars(self._delegate), 'state') + _complex_sites(bound.arguments, 'arguments')
            if self._backend == 'rust' and sites:
                if name not in _SUPPORTED or bound.arguments or any(layout not in ('scalar', 'array', 'masked') for _, layout in sites):
                    raise TypeError(f'Automatic complex state support is not verified for FireSize.{name}; select Python explicitly.')
                from .auto import _warn_complex
                from .components.fire_size import FireSize

                values = copy.deepcopy(vars(self._delegate))
                # Warning-as-error must leave the original delegate untouched.
                _warn_complex('fire_size.' + name, 3)
                delegate = FireSize()
                vars(delegate).update(values)
                object.__setattr__(self, '_delegate', delegate)
                object.__setattr__(self, '_backend', 'python')
            result = getattr(self._delegate, name)(*args, **kwargs)
            self._status['backend'] = self._backend
            return result

        return call

    def __init__(self):
        """Create owned real Rust state without consuming a fallback warning.

        :return: None; initializes the delegate and completion status.
        """
        from .rust import FireSize

        object.__setattr__(self, '_delegate', FireSize())
        object.__setattr__(self, '_backend', 'rust')
        object.__setattr__(self, '_status', self._namespace_status if self._namespace_status is not None else {'backend': 'rust'})

    def __reduce__(self):
        """Serialize complete values and the pinned backend without pointers.

        :return: Stable restoration callable and independent state arguments.
        """
        return _restore_fire_size, (self._backend, copy.deepcopy(vars(self._delegate)))

    def __setattr__(self, name, value):
        """Record caller state edits on the current delegate.

        :param name: Public physical field or private proxy member.
        :param value: Supplied value.
        :return: None; updates state without scientific execution.
        """
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._delegate, name, value)

    @property
    def backend_in_use(self):
        """Report the pinned delegate.

        :return: python or rust.
        """
        return self._backend

    @property
    def requested_backend(self):
        """Retain the original automatic policy.

        :return: auto.
        """
        return 'auto'
