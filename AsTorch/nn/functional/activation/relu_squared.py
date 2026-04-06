import numpy as np
from AsTorch import Tensor
from AsTorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def relu_squared(input):
    mask = Tensor(np.where(input.data < 0, 0, 1).astype(input.dtype))
    relu = mask * input
    return relu ** 2

def manual_relu_squared(input):
    
    input_arr = get_inner_array(input)
    mask = (input_arr < 0)
    input_arr[mask] = 0 # <- inplace replacement of values

    def _relu_squared_backward(output_grad):
        if input.requires_grad:
            
            grad_input = output_grad * np.where(mask, 2 * input_arr, 0)

            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        input_arr**2,
        requires_grad=requires_grad,
        grad_fn=_relu_squared_backward if requires_grad else None,
        grad_fn_name="<ReLUSquaredBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out
