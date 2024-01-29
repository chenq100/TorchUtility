import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import io

class AESCipher:
    """
    A class that provides AES encryption and decryption for files. This class supports:
    - Encrypting data from one file and writing the encrypted data to another file.
    - Decrypting data from one file and writing the decrypted data to another file.
    - Decrypting data from a file and loading the decrypted data into memory.
    """

    DEFAULT_BUFFER_SIZE = 1024 * 1024

    def __init__(self, key=None, password=None):
        """
        Initialize the AES Cipher with either a key or a password.
        Args:
            key (bytes, optional): AES key for encryption/decryption (32 bytes for AES-256).
            password (str, optional): Password to generate a key for encryption/decryption.
        """
        if key:
            self.key = key
        elif password:
            self.key = hashlib.sha256(password.encode()).digest()
        else:
            raise ValueError("Both key and password are None")

    def encrypt_file_to_file(self, input_file_path, output_file_path):
        """
        Encrypts a file and writes the encrypted data to another file.
        Args:
            input_file_path (str): Path to the input file to be encrypted.
            output_file_path (str): Path where the encrypted file will be saved.
        """

    def decrypt_file_to_file(self, input_file_path, output_file_path):
        """
        Decrypts a file and writes the decrypted data to another file.
        Args:
            input_file_path (str): Path to the input file to be decrypted.
            output_file_path (str): Path where the decrypted file will be saved.
        """

    def decrypt_file_to_memory(self, input_file_path, use_disk_buffer=False):
        """
        Decrypts a file and returns an io.BytesIO object with the decrypted data.
        Args:
            input_file_path (str): Path to the file to be decrypted.
            use_disk_buffer (bool): where to temporarily save the decrypted data.
        Returns:
            io.BytesIO: BytesIO object containing the decrypted data.
        """

