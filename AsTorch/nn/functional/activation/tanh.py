import numpy as np
from AsTorch import Tensor
from AsTorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def tanh(input):
    
    input_arr = get_inner_array(input)
    output = np.tanh(input_arr)

    def _tanh_backward(input_grad):
        if input.requires_grad:
            # derivative of tanh(x) = 1 - tanh(x)^2
            grad_input = input_grad * (1 - output ** 2)
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_tanh_backward if requires_grad else None,
        grad_fn_name="<TanhBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out
