import numpy as np
import cupy as cp
from AsTorch import Tensor

class Module:
    def __init__(self):
        
        self._parameters = {}
        self._modules = {} #linear or conv
        self._buffers = {} #ex batch normalization
        self._non_persistent_buffers = set()
        self.training = True  #to know in which mode you are for example dropout change between trin and test
    # built-in Python
    def __setattr__(self, name, value):

        if "_modules" not in self.__dict__:
            if isinstance(value, (Module, Tensor)):
                raise RuntimeError(
                    f"Cannot assign {type(value).__name__} to '{name}' "
                    "before calling super().__init__() in your Module subclass."
                )
            return object.__setattr__(self, name, value)

        # Register parameters
        if isinstance(value, Tensor):
            self._parameters[name] = value

        # Register submodules (including ModuleList)
        elif isinstance(value, Module):
            self._modules[name] = value

        # Always assign normally
        return object.__setattr__(self, name, value)
    # help optimizer to find wait need to update
    def parameters(self, memo=None):

        if memo is None:
            memo = set()

        for param in self._parameters.values():

            if "cuda" in param.device:
                ptr = param.data.data.ptr # <- use cupy array pointer
            else:
                ptr = id(param.data)  # <- use standard id 

            if param is not None and ptr not in memo:
                memo.add(ptr)
                yield param # better than return because it make function work likw Generator
        # back to previous layer and previous and so on 
        for module in self._modules.values():
            yield from module.parameters(memo)
    # it's allow duplicates and we use it if we have shared parameter between layer
    def _parameters_no_dedup(self, prefix=""):
      
        for param in self._parameters.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()
    
    #Same as parameters, but we also return the name layer1.weight', Tensor(10,5) and it's very important for freezing and fine tuning
    def named_parameters(self, prefix="", memo=None):

        if memo is None:
            memo = set()

        for name, param in self._parameters.items():

            if "cuda" in param.device:
                ptr = param.data.data.ptr
            else:
                ptr = id(param.data)

            if param is not None and ptr not in memo:
                memo.add(ptr)
                full = f"{prefix}{name}" if prefix else name
                yield full, param

        for name, m in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from m.named_parameters(sub_prefix, memo)
    # Yield all parameters with names, including duplicates
    def _named_parameters_no_dedup(self, prefix=""):
        
        for name, param in self._parameters.items():
            full_name = f"{prefix}{name}" if prefix else name
            yield full_name, param
        for name, module in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from module._named_parameters_no_dedup(sub_prefix)
    # used to save non training parameter like mask 
    def register_buffer(self, name, tensor, persistent=True):

        if not isinstance(tensor, Tensor):
            raise TypeError("Buffers must be Tensors")
        self._buffers[name] = tensor

        if not persistent:
            self._non_persistent_buffers.add(name)

        object.__setattr__(self, name, tensor)
    # used to retrive all non trainable parameter we have
    def named_buffers(self, prefix="", memo=None, persistent_only=False):
   
        if memo is None:
            memo = set()
        
        for name, buf in self._buffers.items():

            if persistent_only and name in self._non_persistent_buffers:
                continue

            if buf is not None:
                if "cuda" in buf.device:
                    ptr = buf.data.data.ptr
                else:
                    ptr = id(buf.data)
                
                if ptr not in memo:
                    memo.add(ptr)
                    full = f"{prefix}{name}" if prefix else name
                    yield full, buf
        
     
        for name, m in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from m.named_buffers(sub_prefix, memo, persistent_only)
    # it's same to previous but just it's remove duplicate
    def _buffers_no_dedup(self, persistent_only=False):
      
        for name, param in self._buffers.items():
            if persistent_only and name in self._non_persistent_buffers:
                continue
            yield param

        for name, module in self._modules.items():
            yield from module._buffers_no_dedup(persistent_only)
    # same but with we add name
    def _named_buffers_no_dedup(self, prefix="", persistent_only=False):
      
        for name, param in self._buffers.items():
            if persistent_only and name in self._non_persistent_buffers:
                continue
            full_name = f"{prefix}{name}" if prefix else name
            yield full_name, param

        for name, module in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from module._named_buffers_no_dedup(sub_prefix, persistent_only)
   
    # Moves all parameters and buffers of this module to the given device.
    def to(self, device):
       # Move parameters
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = param.to(device)
                object.__setattr__(self, name, self._parameters[name])

        # Move buffers
        for name, buf in self._buffers.items():
            if buf is not None:
                self._buffers[name] = buf.to(device)
                object.__setattr__(self, name, self._buffers[name])

        # Recursively move submodules
        for m in self._modules.values():
            m.to(device)

        return self
    # this help me to apply any thing to all modules
    def apply(self, fn):     

        fn(self)
        for m in self._modules.values():
            m.apply(fn)
        return self
    
    # help me to override it later
    def _extra_repr(self):

        return ""

    def _repr(self, indent=0):
        model_name = self.__class__.__name__
        ind = "   " * indent
        extra = self._extra_repr()
        if not self._modules:  # leaf
            return f"{ind}{model_name}({extra})\n"
        s = f"{ind}{model_name}(\n"
        for key, val in self._modules.items():
            s += f"{ind}  ({key}): {val._repr(indent + 1).lstrip()}"
        s += f"{ind})\n"
        return s

    def __repr__(self):
        return self._repr(indent=0).rstrip()
     
    # my_layer(input_data) <-- my_layer.forward(input_data)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    #Returns a dictionary of all parameters and buffers as NumPy arrays
    def state_dict(self):
        state = {}
        
        # Save all parameters recursively
        for name, param in self._named_parameters_no_dedup():
            if "cuda" in param.device:
                state[name] = param.numpy()
            else:
                state[name] = param.numpy()
        
        # Save all buffers recursively (only want to save persistent buffers)
        for name, buf in self._named_buffers_no_dedup(persistent_only=True):
            if buf is not None:
                if "cuda" in buf.device:
                    state[name] = buf.numpy()
                else:
                    state[name] = buf.numpy()
        
        return state
    
    # load model state and handle if there is any missing value 
    def load_state_dict(self, state_dict, strict=True, device="cpu"):
        
        missing_keys = []
        unexpected_keys = list(state_dict.keys())
        
        # Utility to move arrays to correct backend
        # Default to float32 here
        def to_device(array):
            if device == "cuda":
                return cp.asarray(array, dtype=cp.float32)
            else:
                return array.astype(np.float32)
        
        # Load parameters recursively
        for name, param in self._named_parameters_no_dedup():
            if name in state_dict:
                try:
                    param.data[:] = to_device(state_dict[name])
                except:
                    print(f"Failed to load {name}. Expected {param.shape}, got {state_dict[name].shape}")
                    continue
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)
        
        # Load buffers recursively
        for name, buf in self._named_buffers_no_dedup(persistent_only=True):
            if name in state_dict:
                buf.data[:] = to_device(state_dict[name])
                try:
                    buf.data[:][:] = to_device(state_dict[name])
                except:
                    print(f"Failed to load {name}. Expected {param.shape}, got {state_dict[name].shape}")
                    continue
                if name in unexpected_keys:
                    unexpected_keys.remove(name)
            else:
                missing_keys.append(name)
        
        if strict:
            error_msgs = []
            if missing_keys:
                error_msgs.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                error_msgs.append(f"Unexpected keys: {unexpected_keys}")
            if error_msgs:
                raise RuntimeError("Error(s) in loading state_dict:\n" + "\n".join(error_msgs))
            else:
                return "<All Keys Matched Successfully>"
        else:
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                return "<All Keys Matched Successfully>"
            else:
                return missing_keys, unexpected_keys

    def train(self):
       
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        
        self.training = False
        for m in self._modules.values():
            m.eval()
