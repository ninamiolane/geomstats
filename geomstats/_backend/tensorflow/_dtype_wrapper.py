import functools
import inspect

from tensorflow import cast
from tensorflow.dtypes import as_dtype

_DEFAULT_DTYPE = None

_TO_UPDATE_FUNCS_DTYPE = []


def _update_func_default_dtype(func):
    _TO_UPDATE_FUNCS_DTYPE.append(func)
    return func


def _get_dtype_pos_in_defaults(func):
    pos = 0
    for name, parameter in inspect.signature(func).parameters.items():
        if name == "dtype":
            return pos
        if parameter.default is not inspect._empty:
            pos += 1
    else:
        raise Exception("dtype is not kwarg")


def _update_default_dtypes():
    funcs = _TO_UPDATE_FUNCS_DTYPE
    for func in funcs:
        pos = _get_dtype_pos_in_defaults(func)
        defaults = list(func.__defaults__)
        defaults[pos] = _DEFAULT_DTYPE
        func.__defaults__ = tuple(defaults)


def _cast_fout_from_dtype(dtype_pos=None, _func=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            out = func(*args, **kwargs)

            if out.dtype.is_floating:
                if dtype_pos is not None and len(args) > dtype_pos:
                    dtype = args[dtype_pos]
                else:
                    dtype = kwargs.get("dtype", _DEFAULT_DTYPE)

                if out.dtype != dtype:
                    return cast(out, dtype)

            return out

        return _wrapped

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def set_default_dtype(value):
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = as_dtype(value)
    _update_default_dtypes()

    return get_default_dtype()


def get_default_dtype():
    return _DEFAULT_DTYPE
