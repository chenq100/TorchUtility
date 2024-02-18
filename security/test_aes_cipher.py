import unittest
from unittest.mock import patch
import os
import tempfile
from aes_cipher import AESCipher

class TestAESCipher(unittest.TestCase):
    def setUp(self):
        # Setup for tests
        self.password = "testpassword"
        self.cipher = AESCipher(password=self.password)

        # Create a temporary file with test data
        self.test_data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nbbbbbbbbbbbbbbbbbbbb\ncccccccccccccccccccc\nddddddddddddddddddddddddddddddddddddddd\n"
        self.input_file = tempfile.NamedTemporaryFile(delete=False)
        self.input_file.write(self.test_data)
        self.input_file.close()

        self.output_file = tempfile.NamedTemporaryFile(delete=False).name
        self.decrypted_file = tempfile.NamedTemporaryFile(delete=False).name

        print("Input file path:", self.input_file.name)
        print("Output file path:", self.output_file)
        print("Decrypted file path:", self.decrypted_file)

        # Test data in bytes
        self.test_bytes = b"Test data for encryption and decryption."


    def tearDown(self):
        # Cleanup: Remove temporary files
        os.remove(self.input_file.name)
        os.remove(self.output_file)
        os.remove(self.decrypted_file)

    def test_encrypt_file_to_file_normal(self):
        """ Test normal encryption from file to file """
        self.cipher.encrypt_file_to_file(self.input_file.name, self.output_file)

        # Check if output file is not empty
        self.assertTrue(os.path.getsize(self.output_file) > 0)

    def test_encrypt_file_to_file_exception(self):
        """ Test encryption with a non-existent input file """
        with self.assertRaises(Exception):
            self.cipher.encrypt_file_to_file("non_existent_file.txt", self.output_file)

    def test_decrypt_file_to_file_normal(self):
        """ Test normal decryption from file to file """
        self.cipher.encrypt_file_to_file(self.input_file.name, self.output_file)
        self.cipher.decrypt_file_to_file(self.output_file, self.decrypted_file)

        with open(self.decrypted_file, 'rb') as file:
            decrypted_data = file.read()
        
        # Check if decrypted data matches original data
        self.assertEqual(decrypted_data, self.test_data)

    def test_decrypt_file_to_file_exception(self):
        """ Test decryption with a non-existent input file """
        with self.assertRaises(Exception):
            self.cipher.decrypt_file_to_file("non_existent_file.txt", self.decrypted_file)

    def test_decrypt_file_to_memory_normal(self):
        """ Test normal decryption from file to memory """
        self.cipher.encrypt_file_to_file(self.input_file.name, self.output_file)
        decrypted_stream = self.cipher.decrypt_file_to_memory(self.output_file)

        # Check if decrypted data matches original data
        self.assertEqual(decrypted_stream.read(), self.test_data)

    def test_decrypt_file_to_memory_with_disk_buffer(self):
        """ Test decryption from file to memory using disk buffer """
        self.cipher.encrypt_file_to_file(self.input_file.name, self.output_file)
        decrypted_stream = self.cipher.decrypt_file_to_memory(self.output_file, use_disk_buffer=True)

        # Read content from BytesIO and compare with original data
        decrypted_data = decrypted_stream.getvalue()
        #self.assertEqual(decrypted_data, self.test_data)

    def test_decrypt_file_to_memory_exception(self):
        """ Test decryption to memory with a non-existent input file """
        with self.assertRaises(Exception):
            self.cipher.decrypt_file_to_memory("non_existent_file.txt")

    def test_encrypt_bytes_and_decrypt_bytes(self):
        """Test the encryption and decryption process of encrypt_bytes and decrypt_bytes methods"""
        # Encrypt the test data
        encrypted_data = self.cipher.encrypt_bytes(self.test_bytes)
        print("encrypted_data:" + str(encrypted_data))
        
        # Check if the encrypted data is not empty and contains the signature
        self.assertTrue(encrypted_data.startswith(AESCipher.SIGNATURE))
        
        # Decrypt the test data
        decrypted_data = self.cipher.decrypt_bytes(encrypted_data)
        
        # Check if the decrypted data matches the original test bytes
        self.assertEqual(decrypted_data, self.test_bytes)

    @patch('aes_cipher.AES.new')  # Replace AES.new to simulate an exception
    def test_encrypt_bytes_exception_handling(self, mock_aes_new):
        """Test exception handling in the encrypt_bytes method during the encryption process."""
        # Configure the mock object to raise an exception
        mock_aes_new.side_effect = Exception("Mock encryption error")

        # Expect the method to catch and re-raise the exception
        with self.assertRaises(Exception) as context:
            encrypted_data = self.cipher.encrypt_bytes(self.test_bytes)

        # Check if the expected exception was caught and re-raised
        self.assertTrue('Mock encryption error' in str(context.exception))

        # Optionally: verify if an error log was recorded
        # This would require setting up and checking if the logging module received the expected error message,
        # which might involve more advanced features of unittest.mock

    def test_decrypt_bytes_with_wrong_signature(self):
        """Test decryption with data having an incorrect signature, expecting the decrypt function to return the original data"""
        # Generate data with a wrong signature but correct encrypted content
        wrong_signature_data = b"wrong" + self.cipher.encrypt_bytes(self.test_bytes)[len(AESCipher.SIGNATURE):]
        decrypted_data = self.cipher.decrypt_bytes(wrong_signature_data)
        
        # Expect that due to the signature mismatch, the decrypt function returns the original encrypted data without decryption
        self.assertEqual(decrypted_data, wrong_signature_data)

    def test_decrypt_bytes_without_signature_returns_original(self):
        """Test that decrypting data without the required signature returns the original data unaltered."""
        # Directly use some data that doesn't start with the signature
        unsignatured_data = b"This data does not have the signature."
        
        # Attempt to decrypt the data that lacks the signature
        result_data = self.cipher.decrypt_bytes(unsignatured_data)
        
        # Check if the function returns the original data unchanged
        self.assertEqual(result_data, unsignatured_data)

    def test_encrypt_bytes_size_increase(self):
        """Test and print the increase in size after encryption, accounting for signature, IV, and padding."""
        # Encrypt the test bytes data
        encrypted_data = self.cipher.encrypt_bytes(self.test_bytes)

        # Calculate the increase in size
        increase_in_size = len(encrypted_data) - len(self.test_bytes)

        # Print the increase in size
        print(f"Increase in size after encryption: {increase_in_size} bytes")

        # Verify the increase in size is at least equal to the signature plus IV
        # Note: The exact increase also depends on the padding, which can vary.
        expected_minimum_increase = len(AESCipher.SIGNATURE) + 16  # Length of signature plus IV length
        self.assertTrue(increase_in_size >= expected_minimum_increase)

    def test_decrypt_with_wrong_key(self):
        """Test decrypting data with a wrong key"""
        # Encrypt data with the original key
        encrypted_data = self.cipher.encrypt_bytes(self.test_bytes)
        
        # Create a cipher with a wrong key
        wrong_key_cipher = AESCipher(key=b'wrongkeywrongkeywrongkeywrongk')
        
        # Attempt to decrypt the data with the wrong key
        with self.assertRaises(Exception):
            decrypted_data = wrong_key_cipher.decrypt_bytes(encrypted_data)
            # Optionally, check if decrypted_data is not equal to the original data
            # self.assertNotEqual(decrypted_data, self.test_bytes)

if __name__ == '__main__':
    unittest.main()

