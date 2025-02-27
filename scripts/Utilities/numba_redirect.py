import sys
import logging
import types

logger = logging.getLogger(__name__)


def is_numba_available() -> (bool, Exception):
    """
    Check if numba is available.
    Returns:
        A tuple (available: bool, exception: Exception or None).
    """
    try:
        import numba  # Attempt to import the real numba package.
        return True, None
    except Exception as e:
        return False, e


_available, _exception = is_numba_available()

if _available:
    # Import the real numba and its submodule.
    import numba
    import numba.typed

    # Set sys.modules to the actual numba modules.
    sys.modules["numba"] = numba
    sys.modules["numba.typed"] = numba.typed

else:
    # Provide dummy implementations if numba is not available.
    logger.warning("Numba is not available. Using dummy implementations. Reason: %s", _exception)

    # Create a dummy numba module.
    dummy_numba = types.ModuleType("numba")
    dummy_numba.jit = lambda *args, **kwargs: (lambda f: f)
    dummy_numba.njit = lambda *args, **kwargs: (lambda f: f)
    dummy_numba.prange = range
    dummy_numba.__all__ = ['jit', 'njit', 'prange', 'typed']


    # Define __getattr__ for the dummy numba module to throw errors for undefined attributes.
    def numba_getattr(name):
        raise AttributeError(f"Dummy numba module does not define attribute '{name}'")


    dummy_numba.__getattr__ = numba_getattr

    # Create a dummy submodule for numba.typed.
    dummy_typed = types.ModuleType("numba.typed")
    dummy_typed.List = list  # Use Python's built-in list as a fallback.


    # Define __getattr__ for the dummy numba.typed module.
    def typed_getattr(name):
        raise AttributeError(f"Dummy numba.typed module does not define attribute '{name}'")


    dummy_typed.__getattr__ = typed_getattr

    # Set attributes for the dummy numba module.
    dummy_numba.typed = dummy_typed

    # Insert these dummy modules into sys.modules.
    sys.modules["numba"] = dummy_numba
    sys.modules["numba.typed"] = dummy_typed

__all__ = ['numba_redirect']


def numba_redirect():
    """
      This function exists solely to initialize the script.
      It intentionally does nothing else.
    """
    pass
