"""Read a PyTorch ``.pt`` checkpoint into NumPy arrays without PyTorch.

The MLX runtime has no torch, yet the community real-audio adapters ship as ``torch.save``
zip archives: a ``data.pkl`` pickle whose tensors refer to raw little-endian storages at
``data/<key>``. Only the handful of globals such a checkpoint needs are honoured; anything
else fails to load instead of executing arbitrary pickle code.
"""
from collections import OrderedDict
import pickle
import zipfile

import numpy as np

STORAGE_DTYPES = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
    'BFloat16Storage': np.uint16,  # no NumPy bfloat16: the raw bits are kept
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
}


class _Storage:
    def __init__(self, data, dtype, bfloat16=False):
        self.data, self.dtype, self.bfloat16 = data, dtype, bfloat16


def _rebuild_tensor(storage, offset, size, stride, *_):
    flat = np.frombuffer(storage.data, dtype=storage.dtype)
    size, stride = tuple(int(n) for n in size), tuple(int(n) for n in stride)
    if not size:
        return flat[offset:offset + 1].reshape(())
    contiguous = tuple(int(np.prod(size[i + 1:])) for i in range(len(size)))
    if stride == contiguous:
        return flat[offset:offset + int(np.prod(size))].reshape(size).copy()
    itemsize = flat.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(flat[offset:], shape=size, strides=tuple(s * itemsize for s in stride))
    return np.array(view)


def _rebuild_parameter(data, *_):
    return data


class _Unpickler(pickle.Unpickler):
    GLOBALS = {
        ('torch._utils', '_rebuild_tensor_v2'): _rebuild_tensor,
        ('torch._utils', '_rebuild_parameter'): _rebuild_parameter,
        ('collections', 'OrderedDict'): OrderedDict,
    }

    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if module == 'torch' and name in STORAGE_DTYPES:
            return name
        try:
            return self.GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f'{module}.{name} is not allowed in a weight checkpoint') from None

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == 'storage'):
            raise pickle.UnpicklingError('Unsupported persistent object')
        storage_type, key = pid[1], pid[2]
        with self.archive.open(f'{self.prefix}data/{key}') as f:
            data = f.read()
        return _Storage(data, STORAGE_DTYPES[storage_type], storage_type == 'BFloat16Storage')


def load(path):
    """The checkpoint's Python object, with tensors as NumPy arrays (bfloat16 as raw uint16 bits)."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith('/data.pkl') or n == 'data.pkl']
        if len(pickles) != 1:
            raise ValueError('Not a PyTorch checkpoint archive')
        prefix = pickles[0][:-len('data.pkl')]
        order = archive.read(prefix + 'byteorder').decode().strip() if prefix + 'byteorder' in names else 'little'
        if order != 'little':
            raise ValueError('Big-endian checkpoints are not supported')
        with archive.open(pickles[0]) as f:
            return _Unpickler(f, archive, prefix).load()
