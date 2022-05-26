import ctypes
import numpy as np

type_dict = {
    'void':     None,
    'float':    ctypes.c_float,
    'double':   ctypes.c_double,
    
    'int':      ctypes.c_int,
    'uint':     ctypes.c_uint,
    'int64':    ctypes.c_int64,
    'uint64':   ctypes.c_uint64,
    'uint32':   ctypes.c_uint32,
    'uint32':   ctypes.c_uint32,
    'int16':    ctypes.c_int16,
    'uint16':   ctypes.c_uint16,
    'int8':     ctypes.c_int8,
    'uint8':    ctypes.c_uint8,
    'byte':     ctypes.c_uint8,

    'char*':    ctypes.c_char_p,
    'double*':  np.ctypeslib.ndpointer(dtype='float64', ndim=1, flags='CONTIGUOUS'),
    'float*':   np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='CONTIGUOUS'),
    'int*':     np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='CONTIGUOUS'),
    'short*':   np.ctypeslib.ndpointer(dtype='int16', ndim=1, flags='CONTIGUOUS'),
    'int8*':    np.ctypeslib.ndpointer(dtype='int8', ndim=1, flags='CONTIGUOUS'),
    'uint8*':   np.ctypeslib.ndpointer(dtype='uint8', ndim=1, flags='CONTIGUOUS'),
    'byte*':    np.ctypeslib.ndpointer(dtype='uint8', ndim=1, flags='CONTIGUOUS'),
    '2Dbyte*':  np.ctypeslib.ndpointer(dtype='uint8', ndim=2, flags='CONTIGUOUS'),
    }


class DLL_Loader:
    def __init__(self, path, name):
        self.dll_path = path
        self.dll_name = name
        self.dll = np.ctypeslib.load_library(name, path)
        
    def get_function(self, ret_type, func_name, arg_types):
        func = self.dll.__getattr__(func_name)
        func.restype = type_dict[ret_type]
        func.argtypes = [type_dict[x] for x in arg_types]
        return func
