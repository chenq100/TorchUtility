import unittest
from json_utility import json_cmp 

class TestJSONCmp(unittest.TestCase):
    
    def test_equal_dicts(self):
        dict1 = {"name": "John", "age": 30, "is_student": False}
        dict2 = {"name": "John", "age": 30, "is_student": False}
        self.assertTrue(json_cmp(dict1, dict2))
        
    def test_different_dicts(self):
        dict1 = {"name": "John", "age": 30}
        dict2 = {"name": "Jane", "age": 30}
        self.assertFalse(json_cmp(dict1, dict2))
        
    def test_equal_lists(self):
        list1 = [1, "two", True, None, {"key": "value"}]
        list2 = [1, "two", True, None, {"key": "value"}]
        self.assertTrue(json_cmp(list1, list2))
        
    def test_different_lists(self):
        list1 = [1, "two", True]
        list2 = [1, "two", False]
        self.assertFalse(json_cmp(list1, list2))
        
    def test_different_length_lists(self):
        list1 = [1, "two"]
        list2 = [1, "two", True]
        self.assertFalse(json_cmp(list1, list2))
        
    def test_nested_structures(self):
        dict1 = {"data": {"list": [1, 2, {"key": "value"}]}}
        dict2 = {"data": {"list": [1, 2, {"key": "value"}]}}
        self.assertTrue(json_cmp(dict1, dict2))
        
    def test_different_types(self):
        dict1 = {"key": "value"}
        list1 = ["key", "value"]
        self.assertFalse(json_cmp(dict1, list1))
        
    def test_unordered_lists(self):
        list1 = [1, 2, 3]
        list2 = [3, 2, 1]
        # Assuming your function sorts lists before comparing, this should pass
        self.assertTrue(json_cmp(list1, list2))
        
    def test_unordered_dicts(self):
        # Dictionaries are inherently unordered, but their keys will be compared regardless of order
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}
        self.assertTrue(json_cmp(dict1, dict2))

    def test_deeply_nested_structures(self):
        nested_dict_1 = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": {"level7": {"level8": {"level9": {"level10": "value"}}}}}}}}}}
        nested_list_1 = ["level1", ["level2", ["level3", ["level4", ["level5", ["level6", ["level7", ["level8", ["level9", ["level10", "value"]]]]]]]]]]

        nested_dict_2 = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": {"level7": {"level8": {"level9": {"level10": "different value"}}}}}}}}}}
        nested_list_2 = ["level1", ["level2", ["level3", ["level4", ["level5", ["level6", ["level7", ["level8", ["level9", ["level10", "different value"]]]]]]]]]]


        self.assertFalse(json_cmp(nested_dict_1, nested_dict_2), "The deeply nested dictionaries should not be considered equal due to a difference in the deepest level.")
        self.assertFalse(json_cmp(nested_list_1, nested_list_2), "The deeply nested lists should not be considered equal due to a difference in the deepest level.")

if __name__ == '__main__':
    unittest.main()

