from ..._array import Array
from ...tensor import Tensor

# Tensor --> array
def get_inner_array(arr):
    if isinstance(arr, Array):
        return arr
    elif isinstance(arr, Tensor):
        if hasattr(arr, "data"):
            return arr.data
        else:
            raise Exception("This Tensor object doesn't have a .data attribute!")
# Tensor --> array --> data
def get_inner_inner_array(arr):
    if isinstance(arr, Tensor):
        arr = get_inner_array(arr)
    if hasattr(arr, "_array"):
        return arr._array
    else:
        raise Exception("This Array object doesn't have a ._array attribute!")
