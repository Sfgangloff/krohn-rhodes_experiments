from typing import Set, Dict, Tuple
import itertools

def accepted_words_of_length_n(
    states: Set[str],
    alphabet: Set[str],
    transition: Dict[Tuple[str, str], str],
    start_state: str,
    accept_states: Set[str],
    n: int
) -> Set[str]:
    """
    Compute the set of accepted words of length n for a deterministic finite automaton.

    Parameters:
    - states: set of states Q
    - alphabet: input alphabet Σ
    - transition: transition function δ as a dict {(q, a): q'}
    - start_state: initial state q₀
    - accept_states: set of accepting states F
    - n: desired word length

    Returns:
    - Set of strings of length n accepted by the automaton
    """
    def delta_star(q: str, w: str) -> str | None:
        """Extended transition function δ* from Q × Σ* to Q."""
        for a in w:
            q = transition.get((q, a))
            if q is None:
                return None
        return q

    accepted = set()
    for word_tuple in itertools.product(alphabet, repeat=n):
        word = ''.join(word_tuple)
        final_state = delta_star(start_state, word)
        if final_state in accept_states:
            accepted.add(word)
    return accepted