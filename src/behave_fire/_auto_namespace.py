"""Signature-preserving automatic component routing without native error replay."""
from functools import wraps
import importlib
import inspect
from types import ModuleType

from ._auto_routing import capture_call, choose_backend


def _wrap_function(namespace, name, function):
    """Create one public-signature wrapper around a complete counterpart call.

    :param namespace: Owning routing namespace and actual-backend status.
    :param name: Public attribute name.
    :param function: Unchanged Python counterpart.
    :return: Callable that preflights once before selecting the whole operation.
    """
    operation = namespace._operation_prefix + name

    @wraps(function)
    def routed(*args, **kwargs):
        """Execute one counterpart on its owned original snapshot.

        :param args: Original positional arguments.
        :param kwargs: Original keyword arguments.
        :return: Complete result; exceptions propagate without replay or status fabrication.
        """
        from .auto import _warn_complex

        call = capture_call(operation, function, args, kwargs)
        selected = choose_backend(call, namespace._preferred)
        if selected == 'python' and namespace._preferred == 'rust':
            _warn_complex(operation, 3)
        target = function if selected == 'python' else getattr(namespace._native_namespace(), name)
        result = call.invoke(target)
        namespace._status['backend'] = selected
        return result

    return routed


class AutoNamespace(ModuleType):
    """Automatic namespace with cached preference and shared last-completed status."""

    def __dir__(self):
        """Expose the existing Python public inventory plus routing status.

        :return: Sorted discoverable attribute names without native discovery or execution.
        """
        names = set(dir(self._python_namespace)) | {'backend_in_use', 'preferred_backend'}
        if not self._operation_prefix and self._preferred == 'rust':
            names.update(('CrownGrid', 'SurfaceGrid'))
        return sorted(names)

    def __getattr__(self, name):
        """Resolve a counterpart or explicitly unmigrated class lazily.

        :param name: Public namespace attribute.
        :return: Routed function/submodule, stable registry/enum identity, or selected class.
        """
        if not self._operation_prefix and name == 'BehaveRun':
            from .auto import BehaveRun
            return BehaveRun
        if not self._operation_prefix and name in ('CrownGrid', 'SurfaceGrid'):
            if self._preferred != 'rust':
                raise ImportError(f'{name} is Rust-only; install a compatible native wheel or build the extension.')
            return getattr(self._native_namespace(), name)
        try:
            value = getattr(self._python_namespace, name)
        except AttributeError:
            if self._preferred != 'rust':
                raise
            return getattr(self._native_namespace(), name)
        if isinstance(value, ModuleType) and not name.startswith('_'):
            prefix = self._operation_prefix + name + '.'
            result = AutoNamespace(self._preferred, value, self._native_path + '.' + name,
                                   prefix, self._status)
        elif inspect.isfunction(value) and not name.startswith('_'):
            result = _wrap_function(self, name, value)
        elif inspect.isclass(value) and name in ('FireSize', 'SlopeTool') and self._preferred == 'rust':
            if name == 'FireSize':
                from ._auto_state import fire_size_type
                result = fire_size_type(self._status)
            else:
                result = getattr(self._native_namespace(), name)
        elif inspect.isclass(value) and self._operation_prefix == 'behave_units.' and self._preferred == 'rust':
            # Unit enum identities remain shared; unmigrated converter classes stay explicit.
            result = getattr(self._native_namespace(), name)
        else:
            result = value
        setattr(self, name, result)
        return result

    def __init__(self, preferred, python_namespace=None, native_path='behave_fire.rust', prefix='', status=None):
        """Cache preference without importing a native calculation facade.

        :param preferred: Already probed usable backend name.
        :param python_namespace: Existing Python counterpart module, or the root namespace.
        :param native_path: Matching explicit native module path.
        :param prefix: Canonical component operation prefix.
        :param status: Parent's last-completed backend status shared by child namespaces.
        :return: None; initializes a lazy routing namespace.
        """
        python_namespace = python_namespace or importlib.import_module('behave_fire.python')
        super().__init__(native_path if preferred == 'rust' else python_namespace.__name__)
        self._preferred = preferred
        self._python_namespace = python_namespace
        self._native_path = native_path
        self._operation_prefix = prefix
        self._status = {'backend': preferred} if status is None else status

    def _native_namespace(self):
        """Load the explicitly selected facade only when a real native call needs it.

        :return: Native module; import/facade errors propagate without Python fallback.
        """
        return importlib.import_module(self._native_path)

    @property
    def backend_in_use(self):
        """Report the backend that completed the last direct namespace operation.

        :return: Actual backend name shared across this namespace and its child modules.
        """
        return self._status['backend']

    @property
    def preferred_backend(self):
        """Report the cached backend preference independently of complex fallback.

        :return: Preferred backend name, python or rust.
        """
        return self._preferred
