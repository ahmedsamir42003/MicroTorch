import numpy as np
from AsTorch import Tensor
from AsTorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def leaky_relu(input, negative_slope=0.1):
    mask_pos = Tensor(np.where(input.data > 0, 1, 0).astype(input.dtype))
    mask_neg = 1 - mask_pos
    return input * mask_pos + input * mask_neg * negative_slope

def manual_leaky_relu(input, negative_slope=0.1):
    
    input_arr = get_inner_array(input)
    
    # Compute mask only once and reuse it
    mask = input_arr > 0
    output = np.where(mask, input_arr, negative_slope * input_arr)

    def _leaky_relu_backward(input_grad):
        if input.requires_grad:
            # Reuse the precomputed mask from closure
            grad_input = input_grad * np.where(mask, 1, negative_slope)

            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_leaky_relu_backward if requires_grad else None,
        grad_fn_name="<LeakyReLUBackward>" if requires_grad else None,
        device=input.device,
        dtype=input.dtype,
    )

    if requires_grad:
        out._add_parents(input)

    return out
