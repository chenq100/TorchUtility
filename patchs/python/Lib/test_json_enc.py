import unittest
import json
import os
from patchs.python.Lib.json_enc import enable_json_encryption_patch, disable_json_encryption_patch

class TestJsonEncryption(unittest.TestCase):

    def setUp(self):
        # Setup with a predefined password for AES encryption
        self.test_password = "secure_password"
        enable_json_encryption_patch(self.test_password)
        self.test_file_path = "test_encrypted_json.json"

    def tearDown(self):
        # Make sure to disable the encryption patch after tests
        disable_json_encryption_patch()
        # Clean up the test file
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_encryption_decryption(self):
        # Test that data is correctly encrypted and decrypted
        original_data = {"key": "value", "number": 123}
        encrypted_json = json.dumps(original_data)
        decrypted_data = json.loads(encrypted_json)

        self.assertNotEqual(encrypted_json, json.dumps(original_data))
        self.assertEqual(original_data, decrypted_data)

    def test_disable_patch(self):
        # Test disabling the patch restores original behavior
        original_data = {"key": "value", "number": 123}
        encrypted_json = json.dumps(original_data)
        disable_json_encryption_patch()  # Disable and test original behavior
        regular_json = json.dumps(original_data)

        self.assertNotEqual(encrypted_json, regular_json)
        with self.assertRaises(Exception):
            wrong_data = json.loads(encrypted_json)
            self.assertNotEqual(original_data, wrong_data)
        
        enable_json_encryption_patch(self.test_password)  # Re-enable patch for tearDown cleanup

    def test_write_encrypted_json_to_file(self):
        # Test writing encrypted JSON data to a file
        data = {"message": "This is a test"}
        with open(self.test_file_path, "w") as file:
            json.dump(data, file)  # This should encrypt the data before writing
        
        # Verify that the data in the file is not the plain JSON string
        with open(self.test_file_path, "r") as file:
            file_content = file.read()
            self.assertNotEqual(file_content, json.dumps(data))
 
    def test_encryption_decryption_direct(self):
        # Test that data is correctly encrypted and decrypted without file IO
        original_data = {"key": "value", "number": 123}
        # Use json.dumps to serialize and encrypt data
        encrypted_json = json.dumps(original_data)
        # Use json.loads to deserialize and decrypt data
        decrypted_data = json.loads(encrypted_json)

        self.assertNotEqual(encrypted_json, json.dumps(original_data))
        self.assertEqual(original_data, decrypted_data)
 
    def test_direct_encryption_decryption_with_file(self):
        original_data = {"message": "This is a test"}
        encrypted_json = json.dumps(original_data)

        # Write encrypted data directly to the file
        with open(self.test_file_path, "w") as file:
            file.write(encrypted_json)

        # Read encrypted data directly from the file
        with open(self.test_file_path, "r") as file:
            encrypted_content = file.read()

        decrypted_data = json.loads(encrypted_content)
        self.assertEqual(original_data, decrypted_data)


    def test_read_encrypted_json_from_file(self):
        # Test reading and decrypting JSON data from a file
        data = {"message": "This is a test"}
        with open(self.test_file_path, "w") as file:
            json.dump(data, file)  # Write encrypted data to the file
        
        with open(self.test_file_path, "r") as file:
            decrypted_data = json.load(file)  # This should decrypt the data after reading
        
        self.assertEqual(data, decrypted_data)


if __name__ == "__main__":
    unittest.main()

