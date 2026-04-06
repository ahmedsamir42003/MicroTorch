import numpy as np
from AsTorch import Tensor

from AsTorch.nn.functional.utils import get_inner_inner_array

def gelu(x):
    
    data = x.data

    # Constants
    sqrt_2_over_pi = 0.7978845608 # xp.sqrt(2 / xp.pi).astype(x.data.dtype)
    coeff = 0.044715

    #inner = sqrt_2_over_pi * (x + coeff * x^3)
    x_squared = np.power(data, 2)
    x_cubed = x_squared * data

    inner = sqrt_2_over_pi * (data + coeff * x_cubed)

    ### Tanh out = tanh(inner) ###
    tanh_out = np.tanh(inner)
    out_data = 0.5 * data * (1.0 + tanh_out)

    # Backward
    def _gelu_backward(grad_output):

        if x.requires_grad:
    
            inner_grad = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_squared)

            # derivative of GELU approximation (sech^2(x) = 1 - tanh^2(x))
            sech2 = 1 - np.power(tanh_out, 2)  # derivative of tanh

            grad_input = 0.5 * (1.0 + tanh_out + data * sech2 * inner_grad) * grad_output

            if x.grad is None:
                x.grad = grad_input
            else:
                x.grad += grad_input

    requires_grad = x.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_gelu_backward if requires_grad else None,
        grad_fn_name="<GELUBackward>" if requires_grad else None
    )

    if requires_grad:
        out._add_parents(x)

    return out
