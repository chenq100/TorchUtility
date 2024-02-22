import unittest
import torch
import os
from patchs.pytorch.torch.serialization import enable_encryption_patch, disable_encryption_patch
# Assume your_patch_module contains the enable_encryption_patch and disable_encryption_patch functions

class TestEncryptionFunctionality(unittest.TestCase):
    def setUp(self):
        # Preparation before each test, like defining a test tensor and file path for saving/loading
        self.test_tensor = torch.randn(5, 5)  # Randomly generate a tensor for testing
        self.file_path = 'test_encrypted.pt'  # Define file path for saving and loading

    def tearDown(self):
        # Cleanup after each test, delete the test file
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_encryption_save_load(self):
        # Test the encryption save and load functionality
        enable_encryption_patch('testpassword')  # Enable the encryption patch
        torch.save(self.test_tensor, self.file_path)  # Attempt to save the tensor

        # Load the saved tensor
        loaded_tensor = torch.load(self.file_path)

        disable_encryption_patch()  # Disable the encryption patch to load the data normally

        # Verify if the loaded tensor matches the original tensor
        self.assertTrue(torch.equal(self.test_tensor, loaded_tensor), "The loaded tensor does not match the original tensor.")

if __name__ == '__main__':
    unittest.main()

