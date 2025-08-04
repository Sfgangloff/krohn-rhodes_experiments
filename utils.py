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