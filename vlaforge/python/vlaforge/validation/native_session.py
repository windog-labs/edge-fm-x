"""Python host access to a pinned generated Session through its existing C ABI.

The caller prepares real Torch tensors on their declared devices. All returned
outputs own CPU storage. This module is a Python host, not a no-Python deployment
certificate; standalone native execution and provider audits remain separate.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path
import re
import sys
import threading

from vlaforge.ir.serializer import io_schema_digest, parse_canonical_json
from vlaforge.ir.types import TensorType
from vlaforge.validation.session_benchmark import tensor_bytes


class _Status(C.Structure):
    _fields_ = [('code', C.c_int), ('message', C.c_void_p), ('message_size', C.c_size_t)]


class _Device(C.Structure):
    _fields_ = [('kind', C.c_int), ('ordinal', C.c_int32)]


class _Tensor(C.Structure):
    _fields_ = [('data', C.c_void_p), ('size_bytes', C.c_uint64),
                ('dimensions', C.POINTER(C.c_int64)), ('rank', C.c_uint32),
                ('dtype', C.c_int), ('device', _Device)]


class _BoundTensor(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('tensor', _Tensor),
                ('layout', C.c_int), ('alignment', C.c_uint64)]


class _Stamp(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('has_revision', C.c_uint8),
                ('has_timestamp', C.c_uint8), ('reserved', C.c_uint8 * 6),
                ('revision', C.c_uint64), ('timestamp_ns', C.c_uint64)]


class _ReplayInfo(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('state', C.c_int),
                ('captured_steps', C.c_uint32), ('replay_count', C.c_uint64),
                ('ordinary_count', C.c_uint64), ('reason', C.c_char_p)]


_Bind = C.CFUNCTYPE(_Status, C.c_void_p, C.c_uint32, C.POINTER(_BoundTensor), C.POINTER(_Stamp))
_Run = C.CFUNCTYPE(_Status, C.c_void_p)
_Read = C.CFUNCTYPE(_Status, C.c_void_p, C.c_uint32, C.POINTER(_BoundTensor))
_Destroy = C.CFUNCTYPE(None, C.c_void_p)


class _Api(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('abi_version', C.c_uint32),
                ('schema_digest', C.c_void_p), ('schema_digest_size', C.c_size_t),
                ('bind_tensor', _Bind), ('bind_scalar', C.c_void_p), ('run', _Run),
                ('read_output_tensor', _Read), ('read_output_scalar', C.c_void_p),
                ('reset_episode', C.c_void_p), ('destroy', _Destroy)]


_DTYPES = {'bool': (1, 'bool'), 'i32': (2, 'int32'), 'i64': (3, 'int64'),
           'f16': (4, 'float16'), 'bf16': (5, 'bfloat16'), 'f32': (6, 'float32'),
           'f64': (7, 'float64'), 'u64': (8, 'uint64'), 'u8': (9, 'uint8')}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _check(status, phase):
    if status.code:
        message = (C.string_at(status.message, min(status.message_size, 4096)).decode('utf-8', errors='replace')
                   if status.message else 'no message')
        raise RuntimeError(f'Session {phase} failed ({status.code}): {message}')


def _device(name):
    if name == 'cpu':
        return 0, 0
    if re.fullmatch(r'cuda:[0-9]+', name):
        return 1, int(name.split(':')[1])
    raise ValueError('native host supports only cpu or explicit cuda:N tensor devices')


class NativeTensorSession:
    """Borrow typed input tensors for one synchronous call and return every output.

    One owner process/thread constructs, invokes and closes the Session. Input
    preparation and H2D belong to the caller; run includes binding, native work,
    synchronization, output publication and complete D2H. Inputs must remain
    unchanged until run returns. Every call supplies a fresh monotonic revision.
    No numerical settings, RNG state or device choice are changed implicitly.
    """

    def __init__(self, library, module, *, library_sha256, bundle=None,
                 bundle_sha256=None, cuda_runtime=None):
        if sys.byteorder != 'little' or C.sizeof(C.c_void_p) != 8:
            raise ValueError('native host requires a little-endian 64-bit process')
        library = Path(library).resolve(strict=True)
        if not re.fullmatch('[0-9a-f]{64}', library_sha256) or _sha(library) != library_sha256:
            raise ValueError('native library digest differs')
        if not module.inputs or not module.outputs or any(
            not isinstance(port.payload, TensorType) for port in (*module.inputs, *module.outputs)
        ):
            raise ValueError('native host requires explicit static tensor ports')
        self._specs = {}
        for kind, ports in (('input', module.inputs), ('output', module.outputs)):
            specs = []
            for index, port in enumerate(ports):
                dtype, torch_name = _DTYPES[port.payload.dtype]
                size = tensor_bytes(port.payload.dtype, port.payload.shape)
                identity = getattr(port, kind + '_id')
                specs.append((port, index if identity is None else identity, dtype, torch_name,
                              size, _device(port.device)))
            self._specs[kind] = specs
        if bundle is not None:
            from vlaforge.deployment import load_bundle_manifest

            bundle = Path(bundle).resolve(strict=True)
            if bundle_sha256 is None or _sha(bundle / 'bundle.json') != bundle_sha256:
                raise ValueError('bundle manifest digest differs')
            manifest = load_bundle_manifest(bundle / 'bundle.json')
            manifest.verify_files(bundle)
            if parse_canonical_json((bundle / manifest.semantic_ir.path).read_text()) != module:
                raise ValueError('module differs from the verified bundle')
        elif bundle_sha256 is not None:
            raise ValueError('bundle digest requires a bundle')
        self._owner = os.getpid(), threading.get_ident()
        self._session = C.c_void_p()
        self._revision = 0
        self._inputs = ()
        self.evidence = {'library': str(library), 'library_sha256': library_sha256,
                         'io_schema_digest': io_schema_digest(module),
                         'bundle_sha256': bundle_sha256, 'python_host': True,
                         'standalone_no_python_verified': False}
        self._cuda = None
        ordinals = {device[1] for group in self._specs.values() for *_, device in group if device[0] == 1}
        if len(ordinals) > 1:
            raise ValueError('one Session must use one explicit CUDA device')
        self._cuda_ordinal = next(iter(ordinals), None)
        if ordinals:
            if cuda_runtime is None:
                raise ValueError('CUDA Session requires an explicit CUDA runtime library')
            runtime = Path(cuda_runtime).resolve(strict=True)
            self._cuda = C.CDLL(str(runtime), mode=C.RTLD_LOCAL)
            self._cuda.cudaGetDevice.argtypes, self._cuda.cudaGetDevice.restype = [C.POINTER(C.c_int)], C.c_int
            self._cuda.cudaDeviceSynchronize.argtypes, self._cuda.cudaDeviceSynchronize.restype = [], C.c_int
            self._cuda.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
            self._cuda.cudaMemcpy.restype = C.c_int
            self.evidence.update(cuda_runtime=str(runtime), cuda_runtime_sha256=_sha(runtime))
            self._require_device()
        self._library = C.CDLL(str(library), mode=C.RTLD_LOCAL)
        get_api = self._library.vlaforge_model_session_api
        get_api.argtypes, get_api.restype = [], C.POINTER(_Api)
        pointer = get_api()
        if not pointer:
            raise ValueError('native library returned a null Session ABI')
        self._api = pointer.contents
        if (self._api.struct_size != C.sizeof(_Api) or self._api.abi_version != 2
                or self._api.schema_digest_size != 64 or not self._api.schema_digest
                or C.string_at(self._api.schema_digest, 64).decode('ascii') != io_schema_digest(module)
                or not all((self._api.bind_tensor, self._api.run, self._api.read_output_tensor, self._api.destroy))):
            raise ValueError('native Session ABI or I/O schema differs')
        if bundle is None:
            create = self._library.vlaforge_model_session_create
            create.argtypes, create.restype = [C.POINTER(C.c_void_p)], _Status
            _check(create(C.byref(self._session)), 'create')
        else:
            create = self._library.vlaforge_model_session_create_from_bundle
            create.argtypes, create.restype = [C.c_char_p, C.c_size_t, C.POINTER(C.c_void_p)], _Status
            encoded = os.fsencode(bundle)
            _check(create(encoded, len(encoded), C.byref(self._session)), 'create from bundle')
        if not self._session:
            raise RuntimeError('native Session creation returned null')

    def _require_owner(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('Session belongs to another process/thread')

    def _require_device(self):
        if self._cuda is not None:
            current = C.c_int()
            if self._cuda.cudaGetDevice(C.byref(current)) or current.value != self._cuda_ordinal:
                raise RuntimeError('calling thread CUDA device differs from the Session')

    def _synchronize(self):
        if self._cuda is not None and self._cuda.cudaDeviceSynchronize():
            raise RuntimeError('Session CUDA completion failed')

    def run(self, inputs):
        self._require_owner()
        import torch
        if not self._session:
            raise RuntimeError('Session is closed')
        self._require_device()
        if set(inputs) != {item[0].name for item in self._specs['input']}:
            raise ValueError('inputs must cover exactly the declared tensor ports')
        prepared = []
        for port, identity, dtype, torch_name, size, device in self._specs['input']:
            value = inputs[port.name]
            if (not isinstance(value, torch.Tensor) or value.dtype != getattr(torch, torch_name)
                    or tuple(value.shape) != port.payload.shape or str(value.device) != port.device
                    or not value.is_contiguous() or value.numel() * value.element_size() != size
                    or value.data_ptr() % port.alignment):
                raise ValueError('input dtype/shape/device/storage differs: ' + port.name)
            dimensions = (C.c_int64 * value.ndim)(*value.shape)
            tensor = _BoundTensor(C.sizeof(_BoundTensor),
                _Tensor(value.data_ptr(), size, dimensions, value.ndim, dtype, _Device(*device)),
                0, port.alignment)
            prepared.append((value, dimensions, tensor, identity))
        if self._revision >= 2**64 - 1:
            raise RuntimeError('input revision exhausted')
        self._revision += 1
        stamp = _Stamp()
        stamp.struct_size, stamp.has_revision, stamp.revision = C.sizeof(_Stamp), 1, self._revision
        # Retain descriptors and storage until completion and subsequent rebinding.
        self._synchronize()
        self._inputs = prepared
        for _, _, tensor, identity in prepared:
            _check(self._api.bind_tensor(self._session, identity, C.byref(tensor), C.byref(stamp)), 'bind')
        self._synchronize()
        _check(self._api.run(self._session), 'run')
        self._synchronize()
        result = {}
        for port, identity, dtype, torch_name, size, device in self._specs['output']:
            value = _BoundTensor()
            _check(self._api.read_output_tensor(self._session, identity, C.byref(value)), 'read output')
            tensor = value.tensor
            if (value.struct_size != C.sizeof(_BoundTensor) or value.layout != 0
                    or not tensor.data or tensor.dtype != dtype or tensor.size_bytes != size
                    or tensor.rank != len(port.payload.shape) or (tensor.rank and not tensor.dimensions)
                    or tuple(tensor.dimensions[i] for i in range(tensor.rank)) != port.payload.shape
                    or (tensor.device.kind, tensor.device.ordinal) != device):
                raise RuntimeError('complete output metadata differs: ' + port.name)
            host = torch.empty(port.payload.shape, dtype=getattr(torch, torch_name), device='cpu')
            if device[0] == 1:
                if self._cuda.cudaMemcpy(host.data_ptr(), tensor.data, size, 2):
                    raise RuntimeError('complete output D2H failed')
            else:
                C.memmove(host.data_ptr(), tensor.data, size)
            result[port.name] = host
        return result

    def replay_info(self, task_id):
        """Read an existing compiled loop's counters outside the timed interval."""
        self._require_owner()
        if not self._session:
            raise RuntimeError('Session is closed')
        if type(task_id) is not int or not 0 <= task_id < 2**32:
            raise ValueError('replay task ID must be a uint32 integer')
        self._require_device()
        try:
            read = self._library.vlaforge_model_session_get_replay_info
        except AttributeError as error:
            raise RuntimeError('native library does not expose replay diagnostics') from error
        read.argtypes, read.restype = [C.c_void_p, C.c_uint32, C.POINTER(_ReplayInfo)], _Status
        info = _ReplayInfo()
        info.struct_size = C.sizeof(_ReplayInfo)
        _check(read(self._session, task_id, C.byref(info)), 'read replay info')
        if info.struct_size != C.sizeof(_ReplayInfo) or info.state not in range(4):
            raise RuntimeError('native replay metadata differs')
        return {'task_id': task_id, 'state': info.state, 'captured_steps': info.captured_steps,
                'replay_count': info.replay_count, 'ordinary_count': info.ordinary_count,
                'reason': info.reason.decode('utf-8', errors='replace') if info.reason else None}

    def close(self):
        self._require_owner()
        if self._session:
            self._require_device()
            self._synchronize()
            self._api.destroy(self._session)
            self._session = C.c_void_p()
            self._inputs = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
