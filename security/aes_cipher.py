import io
import os
import zipfile
import tempfile
import hashlib
import logging
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

# Configure basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class AESCipher:
    """
    A class that provides AES encryption and decryption for files. This class supports:
    - Encrypting data from one file and writing the encrypted data to another file.
    - Decrypting data from one file and writing the decrypted data to another file.
    - Decrypting data from a file and loading the decrypted data into memory.
    """

    # Default buffer size for file operations, with the AES block size of 16 bytes.
    #This size determines the maximum amount of plaintext data that might be exposed in the output in case of an exception during encryption or decryption processes. 
    #Larger buffer sizes increase efficiency but also increase the maximum potential exposure of plaintext in case of errors.
    DEFAULT_BUFFER_SIZE = 16 * 1024 * 1024

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
        try:
            with open(input_file_path, 'rb') as infile:
                cipher = AES.new(self.key, AES.MODE_CBC)  # Create a new cipher object
                iv = cipher.iv  # Initialization vector

                with open(output_file_path, 'wb') as outfile:
                    outfile.write(iv)  # Write the IV to the output file
                    while True:
                        data = infile.read(self.DEFAULT_BUFFER_SIZE)
                        if len(data) == 0:
                            break
                        encrypted_data = cipher.encrypt(pad(data, AES.block_size))
                        outfile.write(encrypted_data)
        except Exception as e:
            logging.error(f"Error in encrypt_file_to_file: {e}")
            raise  # Re-raise the exception

    def decrypt_file_to_file(self, input_file_path, output_file_path):
        """
        Decrypts a file and writes the decrypted data to another file.
        Args:
            input_file_path (str): Path to the input file to be decrypted.
            output_file_path (str): Path where the decrypted file will be saved.
        """
        try:
            with open(input_file_path, 'rb') as infile:
                iv = infile.read(16)  # Read the IV from the input file
                cipher = AES.new(self.key, AES.MODE_CBC, iv)  # Create a new cipher object

                with open(output_file_path, 'wb') as outfile:
                    while True:
                        data = infile.read(self.DEFAULT_BUFFER_SIZE)
                        if len(data) == 0:
                            break
                        decrypted_data = cipher.decrypt(data)
                        try:
                            outfile.write(unpad(decrypted_data, AES.block_size))
                        except ValueError:
                            logging.error("Padding error in decrypt_file_to_file")
                            break
        except Exception as e:
            logging.error(f"Error in decrypt_file_to_file: {e}")
            raise  # Re-raise the exception

    def decrypt_file_to_memory(self, input_file_path, use_disk_buffer=False):
        """
        Decrypts a file and returns an io.BytesIO object with the decrypted data.
        If use_disk_buffer is True, uses an encrypted ZIP file on disk to buffer the data.
        Args:
            input_file_path (str): Path to the file to be decrypted.
            use_disk_buffer (bool): Whether to use an encrypted ZIP file on disk for temporary buffering.
        Returns:
            io.BytesIO: BytesIO object containing the decrypted data.
        """
        decrypted_stream = io.BytesIO()
        try:
            if use_disk_buffer:
                # TODO: Implement secure encrypted disk-buffering functionality here
                logging.error("use_disk_buffer functionality is not implemented yet and will be available in future versions.")
                pass
            else:
                with open(input_file_path, 'rb') as infile:
                    iv = infile.read(16)
                    cipher = AES.new(self.key, AES.MODE_CBC, iv)
                    while True:
                        data = infile.read(self.DEFAULT_BUFFER_SIZE)
                        if len(data) == 0:
                            break
                        decrypted_data = cipher.decrypt(data)
                        try:
                            decrypted_stream.write(unpad(decrypted_data, AES.block_size))
                        except ValueError:
                            logging.error("Padding error in decrypt_file_to_memory without disk buffer")
                            break

            decrypted_stream.seek(0)
            return decrypted_stream
        except Exception as e:
            logging.error(f"Error in decrypt_file_to_memory: {e}")
            raise  # Re-raise the exception
