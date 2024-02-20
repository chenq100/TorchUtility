import io
import sys
import torch
import warnings
from typing import Dict
from security.aes_cipher import AESCipher


__all__ = [
    'enable_encryption_patch',
    'disable_encryption_patch'
]

def _save_with_encryption(obj, zip_file, pickle_module, pickle_protocol, _disable_byteorder_record):
    """
    Extends the `torch._save` function by adding encryption to the serialization process. This function
    is intended for internal use by `torch.save` when encryption is required. It serializes the given
    object to the provided zip file, encrypting the content for secure storage.

    This function should be used in conjunction with the `torch.save` function, where encryption parameters
    are specified.

    Args:
        obj (Any): The Python object to be saved. This can be any object that `pickle_module` can serialize.
        zip_file (BinaryIO): The file-like object representing the zip file to which the `obj` is saved.
            This object is expected to encapsulate the file writing logic, supporting write operations
            and possibly format-specific operations for zip archives. It is typically obtained from
            using `_open_zipfile_writer` context manager, which handles the complexities of file
            encoding and PyTorch's internal file writing mechanisms.
            Must support the `write` method.
        pickle_module (module): The module used for serialization, typically `pickle` or a compatible
            module like `dill`.
        pickle_protocol (int): The protocol level to use for pickling. Higher protocol versions are more
            efficient but may not be compatible with older versions of Python.
        _disable_byteorder_record (bool): If True, skips writing the byte order mark that indicates data endianness. 
            Recommended to keep False for data portability across different architectures.

    See Also:
        `torch.save`: The public API function that utilizes this function for saving objects with encryption.
    """
    serialized_storages = {}
    id_map: Dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):

            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = torch.serialization.normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            'Cannot save multiple tensors or storages that '
                            'view the same data as different types')
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            location = torch.serialization.location_tag(storage)
            serialized_storages[storage_key] = storage

            return ('storage',
                    storage_type,
                    storage_key,
                    location,
                    storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    
    if global_aes_cipher is not None:
        # Encrypt the serialized data using the global encryption object
        data_value = global_aes_cipher.encrypt_bytes(data_value)

    zip_file.write_record('data.pkl', data_value, len(data_value))

    # Write byte order marker
    if not _disable_byteorder_record:
        if sys.byteorder not in ['little', 'big']:
            raise ValueError('Unknown endianness type: ' + sys.byteorder)

        zip_file.write_record('byteorder', sys.byteorder, len(sys.byteorder))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in sorted(serialized_storages.keys()):
        name = f'data/{key}'
        storage = serialized_storages[key]
        # given that we copy things around anyway, we might use storage.cpu()
        # this means to that to get tensors serialized, you need to implement
        # .cpu() on the underlying Storage
        if storage.device.type != 'cpu':
            storage = storage.cpu()
        # Now that it is on the CPU we can directly copy it into the zip file
        num_bytes = storage.nbytes()
        zip_file.write_record(name, storage.data_ptr(), num_bytes)

def _load_with_decryption(zip_file, map_location, pickle_module, pickle_file='data.pkl', overall_storage=None, **pickle_load_args):
    """
    Extends the `torch._load` function to add decryption to the deserialization process. This function
    is intended for internal use by `torch.load` when loading objects that were saved with encryption. 
    It loads and decrypts an object from the provided zip file.

    Args:
        zip_file: A file-like object obtained from `torch._open_file_like` function, used for reading 
            the encrypted data. Can handle both actual files and in-memory data structures like io.BytesIO.
        map_location: Specifies how storage locations are remapped. Can be a function, torch.device, 
            a string, or a dict that defines the mapping of location tags to device identifiers.
        pickle_module: The module used for unpickling metadata and objects. Typically `pickle`.
        pickle_file: The name of the file within the zip archive from which to load the serialized data. 
            This file contains the pickled representation of the object to be loaded.
        overall_storage: Optional = None
            An optional storage mechanism used only when `mmap` is enabled. It represents a memory-mapped storage
            of the entire file, allowing efficient and partial data loading by accessing parts of the data on demand
            without loading the entire file into memory. If `mmap` is not enabled, this parameter is not used.
        **pickle_load_args: Additional keyword arguments that will be passed directly to the pickle unpickler during the loading process. 
            This allows for customization of the unpickling process, 
            such as specifying the encoding for text data or handling certain unpickling errors. 
            Useful for advanced scenarios where default unpickling behavior needs to be adjusted.

    Note:
        It's crucial to use the same encryption method and keys for decryption as were used for encryption.
        Ensure the encryption keys are securely managed and accessible at the point of decryption.

    See Also:
        `torch.load`: The public API function that utilizes this function for loading objects with decryption.
    """
    restore_location = torch.serialization._get_restore_location(map_location)

    loaded_storages = {}

    # check if byteswapping is needed
    byteordername = 'byteorder'
    byteorderdata = None
    if zip_file.has_record(byteordername):
        byteorderdata = zip_file.get_record(byteordername)
        if byteorderdata not in [b'little', b'big']:
            raise ValueError('Unknown endianness type: ' + byteorderdata.decode())
    elif torch.serialization.get_default_load_endianness() == LoadEndianness.LITTLE or \
            torch.serialization.get_default_load_endianness() is None:
        byteorderdata = b'little'
    elif torch.serialization.get_default_load_endianness() == LoadEndianness.BIG:
        byteorderdata = b'big'
    elif torch.serialization.get_default_load_endianness() == LoadEndianness.NATIVE:
        pass
    else:
        raise ValueError('Invalid load endianness type')

    if not zip_file.has_record(byteordername) and \
            torch.serialization.get_default_load_endianness() is None and \
            sys.byteorder == 'big':
        # Default behaviour was changed
        # See https://github.com/pytorch/pytorch/issues/101688
        warnings.warn("The default load endianness for checkpoints without a byteorder mark "
                      "on big endian machines was changed from 'native' to 'little' endian, "
                      "to avoid this behavior please use "
                      "torch.serialization.set_default_load_endianness to set "
                      "the desired default load endianness",
                      UserWarning)

    def load_tensor(dtype, numel, key, location):
        name = f'data/{key}'
        if overall_storage is not None:
            storage_offset = zip_file.get_record_offset(name)
            storage = overall_storage[storage_offset:storage_offset + numel]
        else:
            storage = zip_file.get_storage_from_record(name, numel, torch.UntypedStorage)._typed_storage()._untyped_storage
        # swap here if byteswapping is needed
        if byteorderdata is not None:
            if byteorderdata.decode() != sys.byteorder:
                storage.byteswap(dtype)

        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage
        typed_storage = torch.storage.TypedStorage(
            wrap_storage=restore_location(storage, location),
            dtype=dtype,
            _internal=True)

        if typed_storage._data_ptr() != 0:
            loaded_storages[key] = typed_storage

        return typed_storage

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = torch.serialization._maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert typename == 'storage', \
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            nbytes = numel * torch._utils._element_size(dtype)
            typed_storage = load_tensor(dtype, nbytes, key, torch.serialization._maybe_decode_ascii(location))

        return typed_storage

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        'torch.tensor': 'torch._tensor'
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and 'Storage' in name:
                try:
                    return torch.serialization.StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    raw_data = zip_file.get_record(pickle_file)
    if global_aes_cipher is not None:
        # Decrypt the raw data using the global encryption object
        raw_data = global_aes_cipher.decrypt_bytes(raw_data)

    data_file = io.BytesIO(raw_data)

    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    torch._utils._validate_loaded_sparse_tensors()
    torch._C._log_api_usage_metadata(
        "torch.load.metadata", {"serialization_id": zip_file.serialization_id()}
    )
    return result

# Back up the original torch.serialization._save and _load functions
original_torch__save = torch.serialization._save
original_torch__load = torch.serialization._load

# global_aes_cipher is intended for multi-threaded use and is thread-safe,
# The AESCipher instance methods operate solely on input parameters without shared mutable state.
global_aes_cipher = None


def enable_encryption_patch(password):
    """
    Apply Monkey Patching to replace the torch.serialization._save and _load
    functions with custom encryption and decryption functions.
    This enables encryption for torch.save and decryption for torch.load operations.
    """
    global global_aes_cipher
    global_aes_cipher = AESCipher(password=password)
    torch.serialization._save = _save_with_encryption
    torch.serialization._load = _load_with_decryption
    print("Encryption patch enabled.")

def disable_encryption_patch():
    """
    Revoke Monkey Patching by restoring the torch.serialization._save and _load
    functions to their original implementations.
    This disables encryption for torch.save and decryption for torch.load, 
    making them behave as per the default PyTorch functionality.
    """
    torch.serialization._save = original_torch__save
    torch.serialization._load = original_torch__load
    print("Encryption patch disabled.")
