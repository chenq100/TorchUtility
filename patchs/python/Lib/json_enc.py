# cython: language_level=3
import importlib
import json as _standard_json
import base64
from security.aes_cipher import AESCipher

__all__ = [
    'enable_encryption_patch',
    'disable_encryption_patch'
]


def _json_dumps_with_encryption(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw):
    """
    Serialize `obj` to a JSON formatted `str` with encryption, mimicking the original `json.dumps` parameters.
    Interface adapted for Python 3.8 through 3.13 compatibility.
    """
    json_str = original_json_dumps(obj, skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular, allow_nan=allow_nan, cls=cls, indent=indent, separators=separators, default=default, sort_keys=sort_keys, **kw)
    json_data = json_str.encode() # Encode string into binary data, using UTF-8 encoding by default
    encrypted_data = global_aes_cipher.encrypt_bytes(json_data) 
    # Convert encrypted binary data to a Base64-encoded string for safe text-based storage or transport
    encrypted_str = base64.b64encode(encrypted_data).decode()
    return encrypted_str

def _json_loads_with_decryption(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw):
    """
    Deserialize `s` (a `str`, `bytes` or `bytearray` instance containing a JSON document) with decryption, mimicking the original `json.loads` parameters.
    Interface adapted for Python 3.8 through 3.13 compatibility.
    """
    encrypted_data = base64.b64decode(s) 
    decrypted_data = global_aes_cipher.decrypt_bytes(encrypted_data)
    decrypted_str = decrypted_data.decode() # Decode binary data back to string, using UTF-8 encoding by default
    obj = original_json_loads(decrypted_str, cls=cls, object_hook=object_hook, parse_float=parse_float, parse_int=parse_int, parse_constant=parse_constant, object_pairs_hook=object_pairs_hook, **kw)
    return obj


# Back up the original json.dumps and json.loads functions
original_json_dumps = _standard_json.dumps
original_json_loads = _standard_json.loads

# global_aes_cipher is intended for multi-threaded use and is thread-safe,
# The AESCipher instance methods operate solely on input parameters without shared mutable state.
global_aes_cipher = None

def enable_json_encryption_patch(password):
    """Apply monkey patching to replace json.dumps and json.loads with custom encrypted and decrypted versions."""
    global global_aes_cipher
    global_aes_cipher = AESCipher(password=password)
    _standard_json.dumps = _json_dumps_with_encryption
    _standard_json.loads = _json_loads_with_decryption
    print("Json Encryption patch enabled.")

def disable_json_encryption_patch():
    """Revert monkey patching by restoring the original json.dumps and json.loads functions."""
    _standard_json.dumps = original_json_dumps
    _standard_json.loads = original_json_loads
    print("Json Encryption patch disabled.")

