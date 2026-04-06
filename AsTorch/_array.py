import warnings
import numpy as np 

try:
     import cupy as cp
     CUDA_AVAILABLE = True
     NUM_AVALIABLE_GPUS = cp.cuda.runtime.getDeviceCount()
except ImportError:
     cp = None
     CUDA_AVAILABLE = False
     NUM_AVALIABLE_GPUS = 0
     warnings.warn("Cupy is not installed :(")

class Array:
    
    # a+b
    _binary_ufuncs = {
        "__add__": "add", 
        "__radd__": "add",
        "__sub__": "subtract", 
        "__rsub__": "subtract",
        "__mul__": "multiply",
        "__rmul__": "multiply",
        "__truediv__": "true_divide", 
        "__rtruediv__": "true_divide",
        "__floordiv__": "floor_divide", 
        "__rfloordiv__": "floor_divide",
        "__matmul__": "matmul", 
        "__rmatmul__": "matmul",
        "__pow__": "power",
        "__rpow__": "power",
        "__mod__": "remainder", 
        "__rmod__": "remainder",
        "__and__": "bitwise_and",
        "__rand__": "bitwise_and",
        "__or__": "bitwise_or",
        "__ror__": "bitwise_or",
        "__xor__": "bitwise_xor", 
        "__rxor__": "bitwise_xor",
        "__lt__": "less",
        "__le__": "less_equal",
        "__gt__": "greater",
        "__ge__": "greater_equal",
        "__eq__": "equal", 
        "__ne__": "not_equal"
    }

    # a += a
    _inplace_ops = {
        "__iadd__": "add",
        "__isub__": "subtract",
        "__imul__": "multiply",
        "__itruediv__": "true_divide",
        "__ifloordiv__": "floor_divide",
        "__imatmul__": "matmul",
        "__ipow__": "power",
        "__imod__": "remainder",
        "__iand__": "bitwise_and",
        "__ior__": "bitwise_or",
        "__ixor__": "bitwise_xor",
    }
    
    # a = -a
    _unary_ufuncs = {
        "__neg__": "negative",
        "__pos__": "positive",
        "__abs__": "absolute",
        "__invert__": "invert",
    }

    def __init__(self,data,device=None,dtype=None):
        
        # uses device
        if device is not None:       
            if device == "cpu":
                tgt_device="cpu"
                tgt_device_idx=None
            elif "cuda" in device:
                if not CUDA_AVAILABLE:
                    raise RuntimeError("CUDA Not supported, Install Cupy")
                
                tgt_device,tgt_device_idx=self._parse_cuda_str(device_str=device)
                
                if tgt_device_idx +1 > NUM_AVALIABLE_GPUS:
                    raise RuntimeError(f"cuda:{tgt_device_idx} does not exist")             
        else:
            if hasattr(data,"device"):         
                    if isinstance(data.device,str):
                        if "cuda" in data.device:
                            tgt_device,tgt_device_idx=self._parse_cuda_str(device_str=str(data.device))
                        else:
                            tgt_device,tgt_device_idx="cpu",None
                    elif isinstance(data.device,cp.cuda.device.Device):
                        tgt_device = "cuda"
                        tgt_device_idx=data.device.id
            else:
                tgt_device = "cpu"
                tgt_device_idx = None
                
        # data type
        if dtype is None: 
            if hasattr(data,"dtype"):
                current_dtype = str(data.dtype)
                if current_dtype == "float64":
                    dtype="float32"
                elif current_dtype=="int64":
                    dtype = "int32"
                else:
                    dtype=current_dtype
            else:
                dtype="float32"
        else:
            if not isinstance(data,str):
                dtype=str(dtype)
         
        # data it self src --> migration
        if isinstance(data,(np.ndarray,cp.ndarray)):
            self._array=data
        else:
            self._array=np.array(data)
        
        src_dev = "cpu" if isinstance(self._array,np.ndarray) else f"cuda:{self._array.device.id}"
        
        self._array = self._move_array(
            self._array,
            src_dev=src_dev,
            tgt_dev=tgt_device,
            tgt_dev_idx=tgt_device_idx
        )
        
        # update type after migration
        current_dtype=str(self._array.dtype)
        if current_dtype != dtype:
            if "cuda" in tgt_device and CUDA_AVAILABLE :
                with cp.cuda.Device(tgt_device_idx):
                    self._array=self._array.astype(dtype)
            else:
                self._array =self._array.astype(dtype)
                
        # Backend Selection to know how iam 
        self._xp = np if isinstance(self._array,np.ndarray) else cp
        self._dev_id = None if self._xp is np else self._array.device.id
        self._device = "cpu" if self.xp is np else f"cuda:{self._dev_id}"
    
    # it is allow me to handle function just like attr             
    @property
    def xp(self):
        return self._xp
    @property
    def device(self):
        return self._device     
    @property
    def dtype(self):
        return self._array.dtype
    @property
    def shape(self):
        return self._array.shape
    @property
    def ndim(self):
        return self._array.ndim
    @property
    def size(self):
        return self._array.size
    @property
    def T(self):
        return Array(self._array.T, device=self._device)
    
    # update type int,float 32 etc
    def astype(self,dtype):
        
        if self.dtype == dtype:
            return self
       
        if self._xp is np:
            self._array = self._array.astype(dtype)
        else:
            with cp.cuda.Device(self._dev_id):
                self._array = self._array.astype(dtype)
        return self
    
    # build new array on different device
    def to(self,device):
        
        if device == "cuda":
            device= "cuda:0"
        
        if device == self._device:
            return self
        
        else :
            
            if device == "cpu":
                tgt_dev="cpu"
                tgt_dev_idx = None
            else:
                tgt_dev,tgt_dev_idx =self._parse_cuda_str(device)
            
            return Array(data=self._move_array(arr=self._array,
                                               src_dev=self._device,
                                               tgt_dev=tgt_dev,
                                               tgt_dev_idx=tgt_dev_idx),
                         device=device,
                         dtype=self.dtype)
                
    def _parse_cuda_str(self,device_str):
        tgt_device="cuda"
        tgt_device_idx=int(device_str.split(":")[-1]) if ":" in device_str else 0
        return tgt_device,tgt_device_idx
    # this function support Single Responsibility so it's easy to update it latter
    def _move_array(self,arr,src_dev,tgt_dev,tgt_dev_idx=None):
        src_tgt = src_dev if src_dev == "cpu" else "cuda" # we make this because we maybe have more than one gpu
        src_idx = None if src_dev=="cpu" else int(src_dev.split(":")[-1])
        tgt_idx = tgt_dev_idx if tgt_dev == "cuda" else None
        
        if src_tgt == tgt_dev and src_idx==tgt_idx:
            return arr
        
        if tgt_dev == "cuda":
            if not CUDA_AVAILABLE:
                raise RuntimeError("Cuda is not supported :(")
            
            if tgt_dev_idx is None:
                tgt_dev_idx=0
            with cp.cuda.Device(tgt_dev_idx): # choose which card you make process on it
                return cp.asarray(arr)
        else:
            return cp.asnumpy(arr)
    # it's turn any thing to numpy saved run on cpu
    def asnumpy(self):
        if self._device == "cpu":
            return self._array
        return cp.asnumpy(self._array)
    
    # check data type and devic
    def _coerce_other(self, other):
        if isinstance(other, Array):
            return other._array, other._device
        if isinstance(other, np.ndarray):
            return other, "cpu"
        if CUDA_AVAILABLE and isinstance(other, cp.ndarray):
            return other, f"cuda:{other.device.id}"
        return other, None
    
    @classmethod # belong to class not object the function return another one that i will merge to class at first run of program by loop
    def _make_binary_op(cls, ufunc_name, reflect=False):
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            rhs = other_arr
            xp = self._xp
            func = getattr(xp, ufunc_name) # if we pass add -->  xp.add

            if reflect:
                _in = (rhs, self._array)
            else:
                _in = (self._array, rhs)

            if xp is cp:
                with cp.cuda.Device(self._dev_id):
                    res = func(*_in) # * help me to make dispatch
            else:
                res = func(*_in)

            return Array(res, device=self._device)
        
        return op
    
    @classmethod
    def _make_unary_op(cls, ufunc_name):
        def op(self):
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                res = func(self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    res = func(self._array)
            return Array(res, device=self._device)
        return op
    
    @classmethod
    def _make_inplace_op(cls, ufunc_name):
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            # print(other_dev, self._device)
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                func(self._array, other_arr, out=self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    func(self._array, other_arr, out=self._array)
            return self
        return op
    
    def __len__(self):
        return len(self._array)
    
    #print like pytorch
    def __repr__(self):
        data = self._array

        data_str = self.xp.array2string(
            data,
            separator=" ",
            precision=5,
            floatmode="fixed",
            max_line_width=80
        )
        
        lines = data_str.split("\n")
        if len(lines) > 1:
            indent = " " * len("Array(")
            data_str = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

        device_info = f", device='{self.device}'" if "cuda" in self.device else ""

        return f"Array({data_str}, dtype={self.dtype}{device_info})"
    
    # it's help me to run function np.add for example either it's numpy or cupy
    def __array_function__(self, func, types, args, kwargs):
        
        # check if we implement function that need class or no
        if not all(issubclass(t, Array) for t in types):
            return NotImplemented

        devices = set()
        # to handle x,list,tuple,dict
        def handler(x):
            if isinstance(x, Array):
                devices.add(x._device)
                return x._array
            elif isinstance(x, (list, tuple)):
                return type(x)(handler(y) for y in x)
            elif isinstance(x, dict):
                return {k: handler(v) for k, v in x.items()}
            else:
                return x

        handled_args = handler(args)
        handled_kwargs = handler(kwargs)

        if len(devices) > 1:
            raise RuntimeError(f"Expected all tensors to be on the "
                f"same device, but found at least two devices!")

        if not devices:
            device = self._device
        else:
            device = list(devices)[0]

        xp = cp if "cuda" in device else np

        xp_func = getattr(xp, func.__name__, None)

        if xp_func is None:
            return NotImplemented

        if "cuda" in device:
            _, dev_id = self.__parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                result = xp_func(*handled_args, **handled_kwargs)
        else:
            result = xp_func(*handled_args, **handled_kwargs)
        # make sure it's vector else will return scaller
        if isinstance(result, (np.ndarray, cp.ndarray if CUDA_AVAILABLE else type(None))):
            return Array(result, device=device)
        
        return result
    
    # numpy universial function more advanced because it has method like add with cumulative
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        
        arrays = []
        devices = set()

        for x in inputs:
            if isinstance(x, Array):
                arrays.append(x._array)
                devices.add(x._device)
            else:
                arrays.append(x)
                if isinstance(x, np.ndarray):
                    devices.add("cpu")
                elif CUDA_AVAILABLE and isinstance(x, cp.ndarray):
                    devices.add(f"cuda:{x.device.id}")
     
        if len(devices) > 1:
            raise RuntimeError(f"All inputs must be on the same device, found: {devices}")

        device = list(devices)[0] if devices else "cpu"

        if "cuda" in device: 
            _, dev_id = self._parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                result = getattr(ufunc, method)(*arrays, **kwargs)
        else:
            result = getattr(ufunc, method)(*arrays, **kwargs)

        if isinstance(result, (np.ndarray, cp.ndarray)):
            return Array(result, device=device)
        return result

    def __getitem__(self, idx):
    
        def _coerce_index(index):
            if isinstance(index, tuple):
                return tuple(_coerce_index(i) for i in index)
            if isinstance(index, Array):
                return index._array
            if hasattr(index, 'data') and isinstance(index.data, Array):
                return index.data._array
            return index
        
        idx = _coerce_index(idx)
    
        if self.xp == np:
            result = self._array[idx]
        else:
            with cp.cuda.Device(self._array.device.id):
                result = self._array[idx]

        return Array(result, device=self.device)
   
    def __setitem__(self, idx, value):
        if isinstance(value, Array):
            value = value._array
        self._array[idx] = value

    def __getattr__(self, name):      
        if hasattr(self._array, name):
            attr = getattr(self._array, name)
            return attr
        raise AttributeError(f"'Array' object has no attribute '{name}'")
    
    # Wrapping help me to build function that will create functions like .zero , .eye etc
    @classmethod
    def _wrap_factory(cls, xp_func, *args, device="cpu", dtype="float32", **kwargs):
        
        xp = np if "cpu" in device else cp

        _, tgt_device_idx = ("cpu", None)
        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)

        if xp == cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = getattr(xp, xp_func)(*args, **kwargs)
        else:
            arr = getattr(xp, xp_func)(*args, **kwargs)

        if dtype is not None:
            current_dtype = str(arr.dtype)
            if current_dtype != dtype:
                if xp == np:
                    arr = arr.astype(dtype)
                else:
                    with cp.cuda.Device(tgt_device_idx):
                        arr = arr.astype(dtype)

        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def zeros(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("zeros", shape, device=device, dtype=dtype)

    @classmethod
    def ones(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("ones", shape, device=device, dtype=dtype)

    @classmethod
    def empty(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("empty", shape, device=device, dtype=dtype)

    @classmethod
    def full(cls, shape, fill_value, device="cpu", dtype="float32"):
        return cls._wrap_factory("full", shape, fill_value, device=device, dtype=dtype)

    @classmethod
    def eye(cls, N, M=None, k=0, device="cpu", dtype="float32"):
        return cls._wrap_factory("eye", N, M, k, device=device, dtype=dtype)
    
        
    @classmethod
    def tril(cls, x, k=0, device="cpu", dtype="float32"):
        return cls._wrap_factory("tril", x, k=k, device=device, dtype=dtype)
 
    @classmethod
    def zeros_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("zeros_like", other, device=device, dtype=dtype)

    @classmethod
    def ones_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("ones_like", other, device=device, dtype=dtype)

    @classmethod
    def empty_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("empty_like", other, device=device, dtype=dtype)
    
    @classmethod
    def full_like(cls, other, fill_value, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("full_like", other, fill_value, device=device, dtype=dtype)
 
    @classmethod
    def arange(cls, start, end=None, step=1, device="cpu", dtype="float32"):
        if end is None:
            end = start
            start = 0

        return cls._wrap_factory("arange", start, end, step, device=device, dtype=dtype)

    @classmethod
    def linspace(cls, start, end, num=50, device="cpu", dtype="float32"):
        xp = np if "cpu" in device else cp
        arr = xp.linspace(start, end, num=num, dtype=dtype)
        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def randn(cls, shape, device="cpu", dtype="float32"):
        xp = np if "cpu" in device else cp
        tgt_device_idx = None
        
        if "cuda" in device:
           _, tgt_device_idx = cls(None).__parse_cuda_str(device)
        # that is because random genrator is very sensiteve to chip card
        if xp is cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.randn(*shape).astype(dtype)
        else:
            arr = xp.random.randn(*shape).astype(dtype)
        
        return cls(arr, device=device, dtype=str(arr.dtype))
    
    @classmethod
    def rand(cls, shape, device="cpu", dtype="float32"):
        xp = np if "cpu" in device else cp
        tgt_device_idx = None

        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)

        # Generate array on the correct device
        if xp is cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.rand(*shape).astype(dtype)
        else:
            arr = xp.random.rand(*shape).astype(dtype)

        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def randint(cls, low, high, shape, device="cpu", dtype="int32"):
        xp = np if "cpu" in device else cp
        tgt_device_idx = None
        
        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)

        # Generate array on the correct device
        if xp is cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.randint(low, high, size=shape, dtype=dtype)
        else:
            arr = xp.random.randint(low, high, size=shape, dtype=dtype)
  
        return cls(arr, device=device, dtype=str(arr.dtype))
    
    @classmethod
    def randn_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls.randn(other.shape, device=device, dtype=dtype)
    
    @classmethod
    def rand_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls.rand(other.shape, device=device, dtype=dtype)
    
# Attach binary, unary, and inplace operations +,*,-
# setattr(object, name, value)
for dunder, ufunc in Array._binary_ufuncs.items():
    reflect = dunder.startswith("__r")
    setattr(Array, dunder, Array._make_binary_op(ufunc, reflect=reflect))
for dunder, ufunc in Array._unary_ufuncs.items():
    setattr(Array, dunder, Array._make_unary_op(ufunc))
for dunder, ufunc in Array._inplace_ops.items():
    setattr(Array, dunder, Array._make_inplace_op(ufunc))
