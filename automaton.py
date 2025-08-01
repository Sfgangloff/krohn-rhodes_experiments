from typing import Set, Dict, Tuple, Optional,Callable
import yaml

class Automaton:
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
        return self.transitions.get((state, symbol))

    def is_accepting(self, state: str) -> bool:
        return state in self.accepting_states
    
def cascade_automata(A1: Automaton, A2: Automaton, output_map: Callable[[str], str]) -> Automaton:
    """
    Construct the cascade of two automata A1 and A2.

    Parameters:
        A1 (Automaton): First automaton.
        A2 (Automaton): Second automaton, controlled by the output of A1.
        output_map (Callable): A function from A1.state -> A2.input symbol.

    Returns:
        Automaton: The cascade automaton.
    """
    states = {(q1, q2) for q1 in A1.states for q2 in A2.states}
    alphabet = A1.alphabet
    transitions = {}
    initial_state = (A1.initial_state, A2.initial_state)
    accepting_states = {(q1, q2) for q1 in A1.states for q2 in A2.states if A2.is_accepting(q2)}

    for (q1, q2) in states:
        for a in alphabet:
            q1_next = A1.transition(q1, a)
            a2 = output_map(q1_next)
            q2_next = A2.transition(q2, a2)
            transitions[((q1, q2), a)] = (q1_next, q2_next)

    return Automaton(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial_state=initial_state,
        accepting_states=accepting_states
    )

def create_automaton_from_yaml(file:str):
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    states = config["states"]
    alphabet = config["alphabet"]
    initial_state = config["initial_state"]
    accepting_states = config["accepting_states"]

    # Convert transitions back to tuple keys
    transitions = {}
    for k, v in config["transitions"].items():
        src, sym = k.split(",")
        transitions[(src, sym)] = v

    return Automaton(states=states,
                     alphabet=alphabet,
                     initial_state=initial_state,
                     accepting_states=accepting_states,
                     transitions=transitions)
    
if __name__ == "__main__":
    automaton = create_automaton_from_yaml("config.yaml")