import pyarrow as pa
import numpy as np

obj = "test string"

print(np.isscalar(obj))
print(np.dtype(str))
print(np.min_scalar_type(obj))
print(pa.from_numpy_dtype(np.min_scalar_type(obj)))

obj = b"test string"

print(np.isscalar(obj))
print(np.dtype(bytes))
print(np.min_scalar_type(obj))
print(pa.from_numpy_dtype(np.min_scalar_type(obj)))
