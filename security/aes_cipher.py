# cython: language_level=3
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
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s')

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
    #DEFAULT_BUFFER_SIZE = 16

    # Define a unique signature to identify the encrypted data
    SIGNATURE = b'myAESCipher'

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
                        block = infile.read(self.DEFAULT_BUFFER_SIZE)
                        if len(block) == 0: # the end
                            block = pad(b'', AES.block_size)
                            outfile.write(cipher.encrypt(block))
                            break
                        if len(block) < self.DEFAULT_BUFFER_SIZE: # the last block
                            block = pad(block, AES.block_size)
                            outfile.write(cipher.encrypt(block))
                            break
                        else:
                            outfile.write(cipher.encrypt(block))
        except Exception as e:
            logging.error(f"Error in encrypt_file_to_file: {e}")
            raise

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
                    data = b''
                    while True:
                        block = infile.read(self.DEFAULT_BUFFER_SIZE)
                        if len(block) == 0:
                            break
                        data += cipher.decrypt(block)

                    try:
                        # Remove padding and write final data
                        decrypted_data = unpad(data, AES.block_size)
                    except ValueError as e:
                        logging.error(f"Error in decrypt_file_to_file during unpadding: {e}. This may be due to incorrect padding or an incorrect decryption key.")
                        raise ValueError("Decryption failed due to unpadding error. This may indicate incorrect padding or an incorrect key.") from e
                    outfile.write(decrypted_data)

        except Exception as e:
            logging.error(f"Error in decrypt_file_to_file: {e}")
            raise

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

                    data = b''
                    while True:
                        block = infile.read(self.DEFAULT_BUFFER_SIZE)
                        if len(block) == 0:
                            break
                        data += cipher.decrypt(block)

                    try:
                        # Remove padding and write final data
                        decrypted_data = unpad(data, AES.block_size)
                    except ValueError as e:
                        logging.error(f"Error in decrypt_file_to_memory during unpadding: {e}. This may be due to incorrect padding or an incorrect decryption key.")
                        raise ValueError("Decryption failed due to unpadding error. This may indicate incorrect padding or an incorrect key.") from e
                    decrypted_stream.write(decrypted_data)

            decrypted_stream.seek(0)
            return decrypted_stream

        except Exception as e:
            logging.error(f"Error in decrypt_file_to_memory: {e}")
            raise  # Re-raise the exception


    def encrypt_bytes(self, data):
        """
        Encrypts a data segment using AES encryption with a new IV for each encryption.
        Prepends the unique signature and IV to the encrypted data. 
        Handles exceptions during encryption.

        This method is thread-safe, as it only accesses immutable instance variables
            and uses local variables within the method scope.
        """
        try:
            cipher = AES.new(self.key, AES.MODE_CBC)
            iv = cipher.iv
            encrypted_data = cipher.encrypt(pad(data, AES.block_size))
            return self.SIGNATURE + iv + encrypted_data
        except Exception as e:
            logging.error(f"Error in encrypt_bytes: Encryption failed: {e}")
            raise  # Re-raise the current exception

    def decrypt_bytes(self, encrypted_data):
        """
        Decrypts a data segment that was encrypted with AES encryption by this class.
        Handles errors during decryption, such as incorrect padding or corrupted data.

        This method is thread-safe, as it only accesses immutable instance variables
            and uses local variables within the method scope.
        """
        try:
            if encrypted_data.startswith(self.SIGNATURE):
                signature_len = len(self.SIGNATURE)
                iv = encrypted_data[signature_len:signature_len+16]
                encrypted_data_without_iv = encrypted_data[signature_len+16:]
                cipher = AES.new(self.key, AES.MODE_CBC, iv)
                try:
                    decrypted_data = unpad(cipher.decrypt(encrypted_data_without_iv), AES.block_size)
                except ValueError as e:
                    logging.error(f"Error in decrypt_bytes during unpadding: {e}. This may be due to incorrect padding or an incorrect decryption key.")
                    raise ValueError("Decryption failed due to unpadding error. This may indicate incorrect padding or an incorrect key.") from e
                return decrypted_data
            else:
                return encrypted_data

        except Exception as e:
            logging.error(f"Error in decrypt_bytes: Decryption failed: {e}")
            raise  # Re-raise the current exception
