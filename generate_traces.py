"""
generate_traces.py

This module provides tools to simulate the behavior of deterministic finite automata (DFA)
by generating input-output traces based on the automaton's transition function.

Overview:
---------
- Randomly generates input sequences (words) over the DFA's alphabet.
- Simulates execution of each sequence on the DFA and records the state transitions.
- Can be used to produce labeled traces for training or testing RNN models.

Functions:
----------
- random_word: Generate a random word of fixed length over a given alphabet.
- chain_transition: Compute the final state reached after processing a sequence of inputs.
- generate_traces: Generate full traces of DFA execution from random input sequences.

Example:
--------
Run this module directly to generate and print 7 traces of length 10 using a DFA
defined in "config.yaml".

Dependencies:
-------------
- PySAT (unused here but imported for future symbolic extensions)
- automaton.py (DFA definition)
- utils.py (for output formatting)
"""

from pysat.solvers import Solver  # imported but not used yet
from itertools import combinations
from automaton import Automaton, create_automaton_from_yaml
from typing import List
from utils import print_aq_sequences

import random

def random_word(alphabet: List[str], n: int) -> List[str]:
    """
    Generate a random word of length n over the given alphabet.

    Args:
        alphabet (List[str]): List of symbols.
        n (int): Desired word length.

    Returns:
        List[str]: A word of length n (as list of symbols).
    """
    return random.choices(alphabet, k=n)

def chain_transition(inputs: List[str], automaton: Automaton) -> str:
    """
    Compute the final state of the DFA after processing a sequence of input symbols.

    Args:
        inputs (List[str]): Sequence of input symbols.
        automaton (Automaton): DFA to execute the transitions.

    Returns:
        str: Final state reached by the DFA.
    """
    current_state = automaton.initial_state
    for symbol in inputs:
        current_state = automaton.transition(current_state, symbol)
    return current_state

def generate_traces(n: int, automaton: Automaton, num_solutions: int) -> List[List[tuple[str, str]]]:
    """
    Generate random execution traces from the DFA.

    Each trace consists of a list of (symbol, state) pairs,
    representing the transitions taken when reading a random word.

    Args:
        n (int): Length of each random word.
        automaton (Automaton): DFA to simulate.
        num_solutions (int): Number of traces to generate.

    Returns:
        List[List[Tuple[str, str]]]: A list of traces.
    """
    traces = []
    for _ in range(num_solutions):
        word = random_word(list(automaton.alphabet), n)
        trace = []
        current_state = automaton.initial_state
        for symbol in word:
            current_state = automaton.transition(current_state, symbol)
            trace.append((symbol, current_state))
        traces.append(trace)
    return traces

if __name__ == "__main__":
    automaton = create_automaton_from_yaml("config.yaml")
    words = generate_traces(10, automaton, 7)
    print_aq_sequences(words)