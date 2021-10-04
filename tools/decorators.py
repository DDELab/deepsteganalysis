"""
Gently borrowed from an earlier version of pytorch lightning
"""
import numbers
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from pytorch_lightning.utilities.apply_func import apply_to_collection

def _apply_to_inputs(func_to_apply: Callable, *dec_args, **dec_kwargs) -> Callable:
    """
    Decorator function to apply a function to all inputs of a function.

    Args:
        func_to_apply: the function to apply to the inputs
        *dec_args: positional arguments for the function to be applied
        **dec_kwargs: keyword arguments for the function to be applied

    Return:
        the decorated function
    """

    def decorator_fn(func_to_decorate):
        # actual function applying the give function to inputs
        def new_func(*args, **kwargs):
            args = func_to_apply(args, *dec_args, **dec_kwargs)
            kwargs = func_to_apply(kwargs, *dec_args, **dec_kwargs)
            return func_to_decorate(*args, **kwargs)

        return new_func

    return decorator_fn


def _apply_to_outputs(func_to_apply: Callable, *dec_args, **dec_kwargs) -> Callable:
    """
    Decorator function to apply a function to all outputs of a function.

    Args:
        func_to_apply: the function to apply to the outputs
        *dec_args: positional arguments for the function to be applied
        **dec_kwargs: keyword arguments for the function to be applied

    Return:
        the decorated function
    """

    def decorator_fn(function_to_decorate):
        # actual function applying the give function to outputs
        def new_func(*args, **kwargs):
            result = function_to_decorate(*args, **kwargs)
            return func_to_apply(result, *dec_args, **dec_kwargs)

        return new_func

    return decorator_fn


def _convert_to_tensor(data: Any) -> Any:
    """
    Maps all kind of collections and numbers to tensors.

    Args:
        data: the data to convert to tensor

    Return:
        the converted data
    """
    if isinstance(data, numbers.Number):
        return torch.tensor([data])
    # is not array of object
    elif isinstance(data, np.ndarray) and np_str_obj_array_pattern.search(data.dtype.str) is None:
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data

    raise TypeError(f"The given type ('{type(data).__name__}') cannot be converted to a tensor!")


def _convert_to_numpy(data: Union[torch.Tensor, np.ndarray, numbers.Number]) -> np.ndarray:
    """Convert all tensors and numpy arrays to numpy arrays.

    Args:
        data: the tensor or array to convert to numpy

    Return:
        the resulting numpy array
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif isinstance(data, numbers.Number):
        return np.array([data])
    elif isinstance(data, np.ndarray):
        return data

    raise TypeError("The given type ('%s') cannot be converted to a numpy array!" % type(data).__name__)


def _numpy_metric_input_conversion(func_to_decorate: Callable) -> Callable:
    """
    Decorator converting all inputs of a function to numpy

    Args:
        func_to_decorate: the function whose inputs shall be converted

    Return:
        Callable: the decorated function
    """
    return _apply_to_inputs(
        apply_to_collection, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)(func_to_decorate)


def _tensor_metric_output_conversion(func_to_decorate: Callable) -> Callable:
    """
    Decorator converting all outputs of a function to tensors

    Args:
        func_to_decorate: the function whose outputs shall be converted

    Return:
        Callable: the decorated function
    """
    return _apply_to_outputs(_convert_to_tensor)(func_to_decorate)


def _numpy_metric_conversion(func_to_decorate: Callable) -> Callable:
    """
    Decorator handling the argument conversion for metrics working on numpy.
    All inputs of the decorated function will be converted to numpy and all
    outputs will be converted to tensors.

    Args:
        func_to_decorate: the function whose inputs and outputs shall be converted

    Return:
        the decorated function
    """
    # applies collection conversion from tensor to numpy to all inputs
    # we need to include numpy arrays here, since otherwise they will also be treated as sequences
    func_convert_inputs = _numpy_metric_input_conversion(func_to_decorate)
    # converts all inputs back to tensors (device doesn't matter here, since this is handled by BaseMetric)
    func_convert_in_out = _tensor_metric_output_conversion(func_convert_inputs)
    return func_convert_in_out