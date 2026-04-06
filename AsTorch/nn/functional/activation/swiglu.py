from AsTorch import Tensor
from AsTorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from .silu import silu
from .sigmoid import manual_sigmoid

def swiglu(input_a, input_b):
    return input_b * silu(input_a)

def manual_swiglu(input_a, input_b):

    a = get_inner_array(input_a)
    b = get_inner_array(input_b)

    sigmoid_a = manual_sigmoid(input_b)
    silu_a = a * sigmoid_a
    output = b * silu_a

    def _swiglu_backward(grad_out):

        # gradient wrt b:
        # d/db [b * SiLU(a)] = SiLU(a)
        if input_b.requires_grad:
            grad_b = grad_out * silu_a
            if input_b.grad is None:
                input_b.grad = grad_b
            else:
                input_b.grad += grad_b

        # gradient wrt a:
        # d/da [b * (a * sigmoid(a))]
        #
        # First derivative of SiLU(a):
        # SiLU'(a) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
        if input_a.requires_grad:
            dsilu_a = sigmoid_a * (1 + a * (1 - sigmoid_a))
            grad_a = grad_out * dsilu_a * b

            if input_a.grad is None:
                input_a.grad = grad_a
            else:
                input_a.grad += grad_a

    requires_grad = (
        (input_a.requires_grad or input_b.requires_grad)
        and Tensor.build_graph_enabled()
    )

    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_swiglu_backward if requires_grad else None,
        grad_fn_name="<SwiGLUBackward>" if requires_grad else None,
        device=input_a.device,
        dtype=input_a.dtype,
    )

    if requires_grad:
        out._add_parents(input_a, input_b)

    return out
