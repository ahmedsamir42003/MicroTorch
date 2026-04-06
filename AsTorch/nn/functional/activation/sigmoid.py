import numpy as np
from AsTorch import Tensor
from AsTorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def sigmoid(input):
    return 1 / (1 + (-input).exp())

def manual_sigmoid(input):
    
    input_arr = get_inner_array(input)
    output = 1 / (1 + np.exp(-input_arr))
    
    def _sigmoid_backward(input_grad):
        if input.requires_grad:
            # derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            grad_input = input_grad * output * (1 - output)
            
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_sigmoid_backward if requires_grad else None,
        grad_fn_name="<SigmoidBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )
    
    if requires_grad:
        out._add_parents(input)
    
    return out
