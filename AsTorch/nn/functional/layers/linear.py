import os
import numpy as np
from AsTorch import Tensor
from AsTorch.nn.functional.utils import get_inner_array, get_inner_inner_array


def reshape_for_linear(x):

    reshaped = False
    *dims, in_features = x.shape

    if len(dims) > 1:
        reshaped = True

    if reshaped:
        x = x.reshape(np.prod(dims), in_features)
   
    return x, dims, reshaped

# use our autograd
def auto_linear(input, weight, bias=None, *args):


    input, dims, reshaped_flag = reshape_for_linear(input)
    out_features = weight.shape[0]

    output = input @ weight.transpose(-1,-2)
    if bias is not None:
        output = output + bias.reshape(1,-1)
    if reshaped_flag:
        output = output.reshape(*dims, out_features)
 
    return output
# manual for better performance
def manual_linear(input, weight, bias=None, *args):



    input, dims, reshaped_flag = reshape_for_linear(input)
    out_features, in_features = weight.shape

    input_arr = get_inner_array(input)
    weight_arr = get_inner_array(weight).T
    if bias is not None:
        bias_arr = get_inner_array(bias)

    output = np.matmul(input_arr, weight_arr)
    if bias is not None:
        output += bias_arr.reshape(1,-1)

    if reshaped_flag:
        output = output.reshape(*dims, out_features)

    def _linear_backward(grad_output):
        if reshaped_flag:
            grad_output = grad_output.reshape(-1, out_features)

        if weight.requires_grad:
            grad_W = np.matmul(input_arr.T, grad_output)

            if weight.grad is None:
                weight.grad = grad_W.T
            else:
                weight.grad += grad_W.T
            grad_W = None

        if bias is not None and bias.requires_grad:
            grad_b = grad_output.sum(axis=0)
            if bias.grad is None:
                bias.grad = grad_b
            else:
                bias.grad += grad_b
            grad_b = None
        
        if input.requires_grad:
            grad_input = np.matmul(grad_output, weight_arr.T)

            grad_input = grad_input.reshape(*dims, in_features)
            
            if input.grad is None:
                input.grad = grad_input
            else:   
                input.grad += grad_input
            grad_input = None
            
    requires_grad = input.requires_grad or weight.requires_grad or \
                            (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output, 
        requires_grad=requires_grad,
        grad_fn=_linear_backward if requires_grad else None,
        grad_fn_name="<LinearBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output

# Use manual_linear as the default linear function for better performance
linear = manual_linear
