"""
generate_traces.py

This module provides tools to generate valid execution traces of a deterministic finite automaton (DFA)
by encoding its dynamics as a Boolean satisfiability (SAT) problem using the PySAT library.

Given:
    - A DFA
    - A desired trace length n

The script constructs a CNF formula whose satisfying assignments correspond to valid sequences of the form:
    [(a₁, q₀), (a₂, q₁), ..., (aₙ, qₙ₋₁)]

That is, each pair (aᵢ, qᵢ₋₁) represents the automaton reading input aᵢ while in state qᵢ₋₁.

---

Main functions:
    - encode_dfa_trace_to_solver: encode the DFA logic into SAT clauses.
    - enumerate_trace_from_dfa_solver: extract traces from satisfying models.
    - generate_traces: convenience wrapper to get a list of traces.
"""

from pysat.solvers import Solver
from itertools import combinations
from automaton import Automaton, create_automaton_from_yaml
from typing import List
from utils import print_aq_sequences

import random
from typing import List

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

def chain_transition(inputs:List[str],automaton:Automaton):
    current_state = automaton.initial_state
    for k in range(len(inputs)):
        current_state = automaton.transition(state=current_state,symbol=inputs[k])
    return current_state

def generate_traces(n: int, automaton: Automaton, num_solutions: int):
    traces = []
    for k in range(num_solutions):
        word = random_word(alphabet=list(automaton.alphabet),n=n)
        trace = []
        current_state = automaton.initial_state
        for l in range(len(word)):
            current_state = automaton.transition(state=current_state,symbol=word[l])
            trace.append((word[l],current_state))
        traces.append(trace)
    return traces

if __name__ == "__main__":
    automaton = create_automaton_from_yaml("config.yaml")
    words = generate_traces(10, automaton, 7)
    print_aq_sequences(words)