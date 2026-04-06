import math
import numpy as np
from AsTorch import Tensor
from ..base_module import Module
import AsTorch.nn.functional as F

class Linear(Module):

    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 auto=False, 
                 fused=False,
                 act_func=None):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.auto = auto
        self.fused = fused
        self.act_func = act_func
        
        # Initialize weight with uniform distribution
        k = math.sqrt(1 / in_features)
        weight_init = np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.weight = Tensor(weight_init, requires_grad=True)

        if self.bias:
            self.use_bias = True
            bias_init = np.random.uniform(-k, k, (out_features,)).astype(np.float32)
            self.bias = Tensor(bias_init, requires_grad=True)
        else:
            self.use_bias = False
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"
    
    def _extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"
    
    def forward(self, x):
        output = F.linear(x, weight=self.weight, bias=self.bias)
        return output
    