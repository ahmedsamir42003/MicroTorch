from AsTorch import Tensor
from AsTorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from .sigmoid import sigmoid, manual_sigmoid

def silu(input):
    sigmoid_out = sigmoid(input)
    return input * sigmoid_out

def manual_silu(input):
    
    input_arr = get_inner_array(input)
    
    # Compute sigmoid once and reuse it
    sigmoid_output = manual_sigmoid(input)
    
    # SiLU(x) = x * sigmoid(x)
    output = input_arr * sigmoid_output
    
    def _silu_backward(input_grad):
        if input.requires_grad:
            # derivative of silu(x) = sigmoid(x) * (x * (1 - sigmoid(x)) + 1)
            # This can be derived from: d/dx[x * sigmoid(x)]
            grad_input = input_grad * (sigmoid_output * (input_arr * (1 - sigmoid_output) + 1))
            
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_silu_backward if requires_grad else None,
        grad_fn_name="<SiLUBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )
    
    if requires_grad:
        out._add_parents(input)
    
    return out
