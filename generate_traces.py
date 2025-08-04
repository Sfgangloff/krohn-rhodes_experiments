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
from utils import print_aq_sequences

def encode_dfa_trace_to_solver(n: int, automaton: Automaton):
    """
    Encode all valid DFA traces of length n into a SAT solver.

    Each satisfying assignment will correspond to a trace:
        (q₀, a₁, q₁, ..., aₙ, qₙ)

    Constraints ensure:
    - Only one input symbol at each time step
    - Only one state at each time step
    - Transitions respect the DFA definition
    - Initial state is fixed

    Args:
        n (int): Length of the input sequence (number of input symbols).
        automaton (Automaton): The DFA to encode.

    Returns:
        Solver: A PySAT solver instance with all constraints added.
        dict: A variable map (type, index, value) → variable ID
              where 'x' represents input symbols, and 'y' represents states.
    """
    solver = Solver()
    varmap = {}
    varcount = 1

    x_vars = {}  # (i, a): variable representing letter a at position i
    y_vars = {}  # (i, q): variable representing being in state q at time i

    # Define variables
    for i in range(1, n + 1):
        for a in automaton.alphabet:
            x_vars[(i, a)] = varcount
            varmap[('x', i, a)] = varcount
            varcount += 1

    for i in range(n + 1):
        for q in automaton.states:
            y_vars[(i, q)] = varcount
            varmap[('y', i, q)] = varcount
            varcount += 1

    # One input symbol per position
    for i in range(1, n + 1):
        literals = [x_vars[(i, a)] for a in automaton.alphabet]
        solver.add_clause(literals)
        for u, v in combinations(literals, 2):
            solver.add_clause([-u, -v])

    # One state per time step
    for i in range(n + 1):
        literals = [y_vars[(i, q)] for q in automaton.states]
        solver.add_clause(literals)
        for u, v in combinations(literals, 2):
            solver.add_clause([-u, -v])

    # Initial state constraint
    for q in automaton.states:
        solver.add_clause(
            [y_vars[(0, q)]] if q == automaton.initial_state else [-y_vars[(0, q)]]
        )

    # Transition constraints
    for i in range(1, n + 1):
        for q1 in automaton.states:
            for a in automaton.alphabet:
                q2 = automaton.transitions.get((q1, a))
                if q2 is not None:
                    solver.add_clause([
                        -y_vars[(i - 1, q1)],
                        -x_vars[(i, a)],
                        y_vars[(i, q2)]
                    ])

    return solver, varmap

def enumerate_trace_from_dfa_solver(solver, automaton, varmap, num_solutions=7):
    """
    Enumerate valid execution traces of the DFA from a SAT solver.

    Each trace is a list of (input_symbol, source_state) pairs:
        [(a₁, q₀), (a₂, q₁), ..., (aₙ, qₙ₋₁)]

    Args:
        solver (Solver): PySAT solver instance with DFA encoding.
        automaton (Automaton): The DFA used for encoding.
        varmap (dict): Variable map from encoding.
        num_solutions (int): Maximum number of traces to extract.

    Returns:
        List[List[Tuple[str, str]]]: List of traces as (a, q) sequences.
    """
    traces = []

    xvars = [(i, a, var) for (typ, i, a), var in varmap.items() if typ == 'x']
    yvars = [(i, q, var) for (typ, i, q), var in varmap.items() if typ == 'y']
    max_i = max(i for (i, _, _) in xvars)

    while len(traces) < num_solutions and solver.solve():
        model = solver.get_model()
        blocking_clause = []

        word = ['?'] * max_i
        states = ['?'] * max_i

        # Build inverse lookup maps
        xdict = {(i, a): var for (i, a, var) in xvars}
        ydict = {(i, q): var for (i, q, var) in yvars}

        for i in range(1, max_i + 1):
            for a in automaton.alphabet:
                if xdict[(i, a)] in model:
                    word[i - 1] = a
                    blocking_clause.append(-xdict[(i, a)])
                else:
                    blocking_clause.append(xdict[(i, a)])

        for i in range(max_i):
            for q in automaton.states:
                if ydict[(i, q)] in model:
                    states[i] = q
                    blocking_clause.append(-ydict[(i, q)])
                else:
                    blocking_clause.append(ydict[(i, q)])

        trace = list(zip(word, states))
        traces.append(trace)

        solver.add_clause(blocking_clause)

    return traces

def generate_traces(n: int, automaton: Automaton, num_solutions: int):
    """
    High-level function to generate valid traces of a DFA using SAT.

    Args:
        n (int): Length of the input word (number of symbols).
        automaton (Automaton): DFA instance.
        num_solutions (int): Number of traces to return.

    Returns:
        List[List[Tuple[str, str]]]: Traces as (input symbol, state) pairs.
    """
    solver, varmap = encode_dfa_trace_to_solver(n, automaton)
    traces = enumerate_trace_from_dfa_solver(solver, automaton, varmap, num_solutions)
    del solver
    return traces

if __name__ == "__main__":
    automaton = create_automaton_from_yaml("config.yaml")
    words = generate_traces(10, automaton, 7)
    print_aq_sequences(words)