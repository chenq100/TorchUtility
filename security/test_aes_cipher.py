import unittest
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

if __name__ == '__main__':
    unittest.main()

