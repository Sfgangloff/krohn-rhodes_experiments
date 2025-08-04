"""
automaton.py

This module defines the `Automaton` class, a minimal representation of a deterministic finite automaton (DFA),
along with utility functions to:

- Build a DFA from a YAML configuration file
- Compose two DFAs via a cascade construction

Typical usage:
--------------
    from automaton import Automaton, create_automaton_from_yaml

    A = create_automaton_from_yaml("dfa.yaml")
    B = create_automaton_from_yaml("controller.yaml")
    C = cascade_automata(A, B, output_map=lambda q: q[-1])
"""

from typing import Set, Dict, Tuple, Optional, Callable,List
import yaml
from utils import flatten

class Automaton:
    """
    A class representing a deterministic finite automaton (DFA).

    Attributes:
        states (Set[str]): Set of state identifiers.
        alphabet (Set[str]): Set of input symbols.
        transitions (Dict[(str, str), str]): Transition function as a mapping (state, symbol) → state.
        initial_state (str): Initial state of the automaton.
        accepting_states (Set[str]): Set of accepting states.

    Methods:
        transition(state, symbol): Returns the next state given a state and input symbol.
        is_accepting(state): Checks whether a state is accepting.
    """

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transitions: Dict[Tuple[str, str], str],
        initial_state: str,
        accepting_states: Set[str]
    ):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.accepting_states = accepting_states

    def transition(self, state: str, symbol: str) -> Optional[str]:
        """
        Compute the transition from a given state and input symbol.

        Args:
            state (str): Current state.
            symbol (str): Input symbol.

        Returns:
            str or None: Next state if defined, otherwise None.
        """
        return self.transitions.get((state, symbol))

    def is_accepting(self, state: str) -> bool:
        """
        Check if a given state is accepting.

        Args:
            state (str): The state to check.

        Returns:
            bool: True if state is accepting, False otherwise.
        """
        return state in self.accepting_states
    def __str__(self) -> str:
        lines = []

        # States
        lines.append("States: {" + ", ".join([str(state) for state in self.states]) + "}")

        # Initial state
        lines.append("Initial state: " + str(self.initial_state))

        # States
        lines.append("Accepting states: {" + ", ".join([str(state) for state in self.accepting_states]) + "}")

        # Alphabet
        lines.append("Alphabet: {" + ", ".join(sorted([str(letter) for letter in self.alphabet])) + "}")

        # Transitions
        lines.append("Transitions:")
        for (state, symbol) in sorted(self.transitions.keys()):
            target = self.transitions[(state, symbol)]
            lines.append(f"  δ({state}, {symbol}) = {target}")

        return "\n".join(lines)
    
def flatten_dfa_states(dfa: Automaton) -> Automaton:
    """
    Given a DFA whose states are possibly nested tuples, return an equivalent DFA
    with flattened tuple states.

    Args:
        dfa (Automaton): Original DFA with potentially nested tuple states.

    Returns:
        Automaton: A new DFA with flattened tuple states.
    """
    # Flatten all states
    state_map = {s: flatten(s) for s in dfa.states}

    # Flatten transitions
    new_transitions = {
        (state_map[s], a): state_map[t]
        for (s, a), t in dfa.transitions.items()
    }

    new_states = set(state_map.values())
    new_initial = state_map[dfa.initial_state]
    new_accepting = {state_map[s] for s in dfa.accepting_states}

    return Automaton(
        states=new_states,
        alphabet=dfa.alphabet,
        transitions=new_transitions,
        initial_state=new_initial,
        accepting_states=new_accepting
    )

def cascade_automata(A1: Automaton, A2: Automaton, output_map: Callable[[str], str]) -> Automaton:
    """
    Construct the cascade of two automata A1 and A2.

    In the cascade construction, A2 receives input determined by the output of A1.
    The output_map translates the current state of A1 to a symbol for A2.

    Args:
        A1 (Automaton): The first automaton.
        A2 (Automaton): The second automaton, which receives derived input.
        output_map (Callable[[str], str]): A function mapping A1 states to A2 input symbols.

    Returns:
        Automaton: The composed cascade automaton.
    """
    states = {(q1, q2) for q1 in A1.states for q2 in A2.states}
    alphabet = A1.alphabet
    transitions = {}
    initial_state = (A1.initial_state, A2.initial_state)
    accepting_states = {(q1, q2) for q1 in A1.states for q2 in A2.states if A2.is_accepting(q2)}

    for (q1, q2) in states:
        for a in alphabet:
            q1_next = A1.transition(q1, a)
            a2 = output_map(q1)
            q2_next = A2.transition(q2, a2)
            transitions[((q1, q2), a)] = (q1_next, q2_next)

    return Automaton(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial_state=initial_state,
        accepting_states=accepting_states
    )

def cascade_multiple_automata(
    automata: List[Automaton],
    output_maps: List[Callable[[str], str]]
) -> Automaton:
    """
    Constructs a cascade of multiple automata.

    Each automaton A_{i+1} receives input determined by a function applied to the state
    of automaton A_i. The cascade is built from left to right using the cascade_automata function.

    Args:
        automata (List[Automaton]): A list of Automaton objects [A1, A2, ..., An].
        output_maps (List[Callable[[str], str]]): A list of output maps [f1, f2, ..., f_{n-1}],
            where f_i maps states of A_i to symbols for A_{i+1}.

    Returns:
        Automaton: The composed cascade automaton.
    
    Raises:
        ValueError: If the number of output_maps is not one less than the number of automata.
    """
    if len(output_maps) != len(automata) - 1:
        raise ValueError("Number of output maps must be one less than the number of automata.")

    current = automata[0]
    for i in range(1, len(automata)):
        current = cascade_automata(current, automata[i], output_maps[i - 1])

    return current

def create_automaton_from_yaml(file: str) -> Automaton:
    """
    Load an Automaton from a YAML file.

    The YAML file must define:
        - states: list of state names
        - alphabet: list of input symbols
        - transitions: a dictionary with string keys like "q,a" and string values (target states)
        - initial_state: the start state
        - accepting_states: list of accepting states

    Example YAML format:
        states: ["0", "1"]
        alphabet: ["0", "1"]
        transitions:
            "0,0": "0"
            "0,1": "1"
            "1,0": "1"
            "1,1": "0"
        initial_state: "0"
        accepting_states: ["0"]

    Args:
        file (str): Path to the YAML file.

    Returns:
        Automaton: The constructed automaton.
    """
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    states = set(config["states"])
    alphabet = set(config["alphabet"])
    initial_state = config["initial_state"]
    accepting_states = set(config["accepting_states"])

    transitions = {}
    for k, v in config["transitions"].items():
        src, sym = k.split(",")
        transitions[(src.strip(), sym.strip())] = v.strip()

    return Automaton(states, alphabet, transitions, initial_state, accepting_states)

def create_automaton_from_dict(data: dict) -> Automaton:
    states = set(data["states"])
    alphabet = set(data["alphabet"])
    transitions = {
        tuple(k.split(",")): v for k, v in data["transitions"].items()
    }
    return Automaton(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial_state=data["initial_state"],
        accepting_states=set(data["accepting_states"])
    )

def load_automata_and_output_maps(config_path: str) -> Tuple[List[Automaton], List[Callable[[str], str]]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    automata_data = config["automata"]
    automata = [create_automaton_from_dict(data) for data in automata_data]

    output_maps = []
    for data in automata_data[:-1]:  # all except last
        mapping = data["output_map"]
        output_maps.append(lambda q, m=mapping: m[q])  # default capture

    return automata, output_maps

if __name__ == "__main__":
    automaton = create_automaton_from_yaml("config.yaml")
    print("Loaded DFA with", len(automaton.states), "states")
    print(automaton)