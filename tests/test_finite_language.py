import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_language import finite_language_from_sequences,FiniteLanguage


class TestFiniteLanguage(unittest.TestCase):

    def test_valid_language(self):
        levels = [
            {""},
            {"a", "b"},
            {"aa", "ab", "ba"},
            {"aab", "aba", "baa"}
        ]
        alphabet = {"a", "b"}
        try:
            lang = FiniteLanguage(levels, alphabet)
        except ValueError:
            self.fail("FiniteLanguage raised ValueError unexpectedly on valid input.")

    def test_forward_extension_violation(self):
        # 'a' ∈ L1 but no word in L2 extends it
        levels = [
            {""},
            {"a", "b"},
            {"bb"}
        ]
        alphabet = {"a", "b"}
        with self.assertRaises(ValueError) as context:
            _ = FiniteLanguage(levels, alphabet)
        self.assertIn("does not extend", str(context.exception))

    def test_backward_compatibility_violation(self):
        # 'bc' ∈ L2, but 'b' ∉ L1
        levels = [
            {""},
            {"a"},
            {"aa", "ac", "bc"}
        ]
        alphabet = {"a", "b", "c"}
        with self.assertRaises(ValueError) as context:
            _ = FiniteLanguage(levels, alphabet)
        self.assertIn("has prefix", str(context.exception))

    def test_empty_language(self):
        levels = [
            {""}
        ]
        alphabet = set()
        lang = FiniteLanguage(levels, alphabet)
        self.assertEqual(lang.levels[0], {""})
        self.assertEqual(lang.alphabet, set())

    def test_repr_output(self):
        levels = [
            {""},
            {"a"},
            {"aa"}
        ]
        alphabet = {"a"}
        lang = FiniteLanguage(levels, alphabet)
        self.assertIn("FiniteLanguage", repr(lang))
        self.assertIn("levels", repr(lang))

class TestFiniteLanguageFromSequences(unittest.TestCase):

    def test_basic_prefix_construction(self):
        strings = ["ababa", "abaab"]
        lang = finite_language_from_sequences(strings)

        expected_levels = [
            {""},
            {"a"},
            {"ab"},
            {"aba"},
            {"abab", "abaa"},
            {"ababa", "abaab"}
        ]

        self.assertEqual(len(lang.levels), 6)
        for i, expected in enumerate(expected_levels):
            self.assertEqual(lang.levels[i], expected)

    def test_single_string(self):
        strings = ["abc"]
        lang = finite_language_from_sequences(strings)

        expected_levels = [
            {""},
            {"a"},
            {"ab"},
            {"abc"}
        ]
        self.assertEqual(len(lang.levels), 4)
        for i in range(4):
            self.assertEqual(lang.levels[i], expected_levels[i])

    def test_alphabet_inference(self):
        strings = ["ab", "ba"]
        lang = finite_language_from_sequences(strings)
        self.assertEqual(lang.alphabet, {"a", "b"})

    def test_well_formedness(self):
        strings = ["xyx", "xxy"]
        lang = finite_language_from_sequences(strings)
        # If construction is valid, this should not raise
        lang._check_well_formedness()

    def test_error_on_unequal_length(self):
        strings = ["abc", "ab"]
        with self.assertRaises(AssertionError):
            _ = finite_language_from_sequences(strings)

if __name__ == "__main__":
    unittest.main()