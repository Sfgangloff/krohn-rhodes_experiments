import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import split_last_letter


class TestSplitLastLetter(unittest.TestCase):

    def test_basic_case(self):
        input_data = ["abc", "x", "hello"]
        expected = [("ab", "c"), ("", "x"), ("hell", "o")]
        self.assertEqual(split_last_letter(input_data), expected)

    def test_single_char_strings(self):
        input_data = ["a", "b", "c"]
        expected = [("", "a"), ("", "b"), ("", "c")]
        self.assertEqual(split_last_letter(input_data), expected)

    def test_mixed_lengths(self):
        input_data = ["hi", "world", "z"]
        expected = [("h", "i"), ("worl", "d"), ("", "z")]
        self.assertEqual(split_last_letter(input_data), expected)

    def test_empty_input_list(self):
        input_data = []
        expected = []
        self.assertEqual(split_last_letter(input_data), expected)

    def test_raises_on_empty_string(self):
        input_data = ["abc", "", "de"]
        with self.assertRaises(ValueError):
            split_last_letter(input_data)


if __name__ == "__main__":
    unittest.main()