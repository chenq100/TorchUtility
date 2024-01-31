

def json_cmp(jdata1, jdata2):
    """
    Compare two JSON data structures (either dictionaries or lists) for deep equality.
    It recursively compares dictionaries and lists, including nested structures,
    and prints the differences if any. The comparison includes checking whether
    both structures have the same type, same keys (for dictionaries), and same values
    (including nested structures). For lists, it sorts them before comparison to handle
    unordered lists assuming the order does not matter for the comparison.

    Parameters:
    - jdata1: The first JSON data structure to compare. Can be a dictionary or list.
    - jdata2: The second JSON data structure to compare. Can be a dictionary or list.

    Returns:
    - True if both JSON data structures are equal (deeply), False otherwise.

    Note:
    - This function prints the path and nature of differences if it finds any,
      making it easier to understand where and how the two JSON structures differ.
    - It assumes that the order of elements in lists does not matter for the
      comparison, hence sorts lists before comparing them.
    - For simplicity, this function does not handle cases where JSON data types
      might be mixed within a list (e.g., comparing a list of dictionaries to a list
      of lists) beyond the basic type checks and comparisons.
    """
    def compare_dicts(dict1, dict2, path=""):
        """
        Compare two dictionaries recursively and print differences with the path.

        :param dict1: First dictionary to compare.
        :param dict2: Second dictionary to compare.
        :param path: Current path to the element being compared, for tracking nested structures.
        :return: True if dictionaries are equal, False otherwise.
        """
        if dict1.keys() != dict2.keys():
            print(f"Diff at {path}: Keys differ. {dict1.keys()} != {dict2.keys()}")
            return False

        for key in dict1.keys():
            current_path = f"{path}.{key}" if path else key
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                if not compare_dicts(dict1[key], dict2[key], current_path):
                    return False
            elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
                if not compare_lists(dict1[key], dict2[key], current_path):
                    return False
            elif dict1[key] != dict2[key]:
                print(f"Diff at {current_path}: [{dict1[key]}] != [{dict2[key]}]")
                return False

        return True

    def compare_lists(list1, list2, path):
        """
        Compare two lists recursively and print differences with the path.

        :param list1: First list to compare.
        :param list2: Second list to compare.
        :param path: Current path to the element being compared, for tracking nested structures.
        :return: True if lists are equal, False otherwise.
        """
        if len(list1) != len(list2):
            print(f"Diff at {path}: Length of lists differ [{len(list1)}] != [{len(list2)}]")
            return False

        # Sorting the lists by a string representation to handle unordered lists
        sorted_list1 = sorted(list1, key=lambda x: str(x))
        sorted_list2 = sorted(list2, key=lambda x: str(x))

        for index, (item1, item2) in enumerate(zip(sorted_list1, sorted_list2)):
            current_path = f"{path}[{index}]"
            if isinstance(item1, dict) and isinstance(item2, dict):
                if not compare_dicts(item1, item2, current_path):
                    return False
            elif item1 != item2:
                print(f"Diff at {current_path}: {item1} != {item2}")
                return False

        return True

    """
    JSON Data Types

    Basic Types:
    1. String: A sequence of characters surrounded by double quotes. Example: "Hello World"
    2. Number: An integer or a floating-point number. Example: 123, 3.14
    3. Boolean: Represents a logical entity and can have two values: true or false. Example: true
    4. Null: Represents a null value. Example: null

    Composite Types:
    1. Objects (Dictionaries): 
      - A collection of key-value pairs.
      - Keys are always strings, and values can be any valid JSON data type (including basic types, objects, or arrays).
      - Enclosed in curly braces {}. 
      - Example: {"name": "John", "age": 30}

    2. Arrays (Lists):
      - An ordered list of values.
      - Each element in an array can be of any JSON data type.
      - Enclosed in square brackets [].
      - Elements are separated by commas.
      - Example: [1, "two", true, null, {"key": "value"}]

    Note: 
      In JSON (JavaScript Object Notation) format, the root level of the structure must be a composite type:
      In JSON objects (dictionaries), each element is a key-value pair, where the key is a string and the value is any JSON data type. 
      In JSON arrays (lists), each element is a single entity and can be of any type, including other arrays or objects.
      Single primitive data types such as a standalone string, number, boolean, or null are not valid as the root level in JSON.
    """
    # Determine the type of the JSON data and call the appropriate function
    if isinstance(jdata1, dict) and isinstance(jdata2, dict):
        return compare_dicts(jdata1, jdata2)
    elif isinstance(jdata1, list) and isinstance(jdata2, list):
        return compare_lists(jdata1, jdata2, "")
    else:
        # Handle the case where the top-level types are different
        print(f"Top-level types differ: {type(jdata1)} is not {type(jdata2)}")
        return False

