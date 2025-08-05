from typing import List, Tuple

def print_aq_sequences(sequences: List[List[Tuple[str, str]]]):
    """
    Given a list of sequences of (a, q) pairs, print:
    - the sequence of a's (inputs) in the first row
    - the sequence of q's (states) in the second row
    """
    for seq in sequences:
        inputs = ' '.join(a for a, _ in seq)
        states = ' '.join(q for _, q in seq)
        print(inputs)
        print(states)
        print()  # Blank line between sequences

def flatten(state) -> Tuple[str, ...]:
    """
    Recursively flatten a tuple of tuples (or strings) into a flat tuple of strings.

    Example:
        flatten((('q0', 'x'), 'q1')) == ('q0', 'x', 'q1')
    """
    if isinstance(state, tuple):
        flat = ()
        for s in state:
            flat += flatten(s)
        return flat
    else:
        return (str(state),)
    
from typing import Optional, Dict

def find_close_key(target: float, d: Dict[str, float], eps: float) -> Optional[str]:
    """
    Search for a key in the dictionary whose value is within eps of the target float.

    Args:
        target (float): The target value to compare against.
        d (Dict[str, float]): Dictionary with string keys and float values.
        eps (float): Allowed distance threshold.

    Returns:
        Optional[str]: The first key whose value is within eps of target, or None if none found.
    """
    for k, v in d.items():
        if abs(v - target) < eps:
            return k
    return None