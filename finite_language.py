from typing import List, Set, Dict, Tuple
from extracting import get_hidden_state
from rnn_last_state_predictor import RNNPredictor
import torch

# TODO: Finish this: given a language to feed to the RNN, we want to build the "hidden automaton". Also there may be better ways to 
# build this automaton ? Also write tests for these functions and extracting. 

class FiniteLanguage:
    def __init__(self, levels: List[Set[str]], alphabet: Set[str]):
        """
        Initialize a FiniteLanguage object.

        Args:
            levels (List[Set[str]]): List of sets, where levels[n] contains words of length n.
            alphabet (Set[str]): The finite alphabet Σ.
        """
        self.levels = levels
        self.alphabet = alphabet
        self._check_well_formedness()
        self._compute_transitions()

    def _check_well_formedness(self):
        """
        Check the consistency conditions:
        1. Forward extension: every word in L_n extends to some word in L_{n+1}
        2. Backward compatibility: every word in L_{n+1} has prefix in L_n
        3. Every word in L_n has length n.
        """
        for n in range(len(self.levels) - 1):
            Ln = self.levels[n]
            Ln1 = self.levels[n + 1]

            # Condition 1: Forward extension
            for w in Ln:
                if not any((w + a) in Ln1 for a in self.alphabet):
                    raise ValueError(f"Word '{w}' in L_{n} does not extend to any word in L_{n+1}.")

            # Condition 2: Backward compatibility
            for w in Ln1:
                prefix = w[:-1]
                if prefix not in Ln:
                    raise ValueError(f"Word '{w}' in L_{n+1} has prefix '{prefix}' not in L_{n}.")
                
            # Condition 3: Length n
            for w in Ln:
                if len(w) != n: 
                    raise ValueError(f"Word '{w}' in L_{n} has not length {n}.")

    # def _compute_transitions(self) -> Dict[str, Dict[str, str]]:
    #     """
    #     For each word s in the language and each symbol a in the alphabet,
    #     find the unique extension s' = s + a in the next level (if it exists).

    #     Returns:
    #         Dict[str, Dict[str, str]]: Outer dict maps words s to inner dicts;
    #         inner dict maps symbol a to word s' = s + a.
    #     """
    #     D = {}


    #     for n in range(len(self.levels) - 1):
    #         Ln = self.levels[n]
    #         Ln1 = self.levels[n + 1]

    #         for s in Ln:
    #             D[s] = {}
    #             for a in self.alphabet:
    #                 extended = s + a
    #                 if extended in Ln1:
    #                     D[s][a] = extended
    #     self.transitions = D
    def __repr__(self):
        return f"FiniteLanguage(levels={self.levels})"
    

# def extract_language_hidden_states(
#     language: FiniteLanguage,
#     model: RNNPredictor,
#     symbol_to_idx: Dict[str, int]
# ) -> Dict[str, torch.Tensor]:
#     """
#     For each word s in the finite language, compute its RNN hidden state.

#     Args:
#         language (FiniteLanguage): The structured language object.
#         model (RNNPredictor): The trained model.
#         symbol_to_idx (dict): Mapping from symbols to integer indices.

#     Returns:
#         Dict[str, torch.Tensor]: Dictionary mapping string words to their hidden state vectors.
#     """
#     D = {}
#     for level in language.levels:
#         for s in level:
#             prefix = list(s)  # split the string into symbols
#             h = get_hidden_state(model, prefix, symbol_to_idx)
#             D[s] = h
#     return D

def build_dfa_from_language(
    language: FiniteLanguage,
    model: RNNPredictor,
    symbol_to_idx: Dict[str, int],
    eps: float
) -> Dict[str, Tuple[torch.Tensor, List[str], Dict[str, Dict[str, str]]]]:
    """
    Build a DFA-like structure from a finite language and an RNN.

    Returns:
        A dictionary D where:
        - Keys are strings (words in the language),
        - Each value is a tuple (h, L, T), where:
            - h: hidden state (torch.Tensor),
            - L: list of associated words,
            - T: transitions {prefix -> {symbol -> word}}
    """
    D: Dict[str, Tuple[torch.Tensor, List[str], Dict[str, Dict[str, str]]]] = {}
    T: Dict[str, Dict[str, str]] = {}
    L: List[str] = []

    for level in language.levels:
        for w in level:
            prefix = w[:-1] if w else ''
            h = get_hidden_state(model, list(w), symbol_to_idx)

            # Case 1: Prefix already known
            if prefix in L:
                L.append(w)
                continue

            # Case 2: Check for similar state in D
            found = False
            for s in D:
                h_s = D[s][0]
                if torch.norm(h - h_s) < eps:
                    # Found match
                    if prefix not in T:
                        T[prefix] = {}
                    T[prefix][w[-1]] = s
                    found = True
                    break

            if found:
                continue

            # Case 3: New state
            D[w] = (h, [], T.copy())  # make a copy of transitions up to now
            if prefix not in T:
                T[prefix] = {}
            T[prefix][w[-1]] = w

    return D
    
if __name__ == "__main__":
    alphabet = {'a', 'b'}

    levels = [
        {''},                        # L_0: empty string
        {'a', 'b'},                  # L_1
        {'aa', 'ab', 'ba', 'bb'},          # L_2
        {'aaa','aab','aba', 'abb', 'baa', 'bab','bba','bbb'} # L_3
    ]

    flang = FiniteLanguage(levels, alphabet)
    print(flang)
    print(flang.transitions)