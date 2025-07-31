from pysat.solvers import Solver
from itertools import combinations
from automaton import Automaton,create_automaton_from_yaml
from utils import print_aq_sequences

def encode_dfa_trace_to_solver(n: int, automaton: Automaton):
    """
    Create a PySAT solver whose CNF encodes all valid execution traces (q_0,a_1,q_1,...,a_n,q_n)
    of the given DFA of length n. No constraint on the final state being accepting.

    Returns:
        solver (Solver)
        varmap (dict): maps (type, position, value) to variable ID
    """
    solver = Solver()
    varmap = {}
    varcount = 1

    x_vars = {}  # (i, a): input symbol a at time i
    y_vars = {}  # (i, q): state q at time i

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

    # One symbol per position
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

def enumerate_trace_from_dfa_solver(solver, automaton,varmap, num_solutions=7):
    traces = []

    xvars = [(i, a, var) for (typ, i, a), var in varmap.items() if typ == 'x']
    yvars = [(i, q, var) for (typ, i, q), var in varmap.items() if typ == 'y']
    max_i = max(i for (i, _, _) in xvars)

    while len(traces) < num_solutions and solver.solve():
        model = solver.get_model()
        blocking_clause = []

        word = ['?'] * max_i
        states = ['?'] * max_i

        # Build inverse maps
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

def generate_traces(n:int,automaton:Automaton,num_solutions:int):
    solver, varmap = encode_dfa_trace_to_solver(n, automaton)
    words = enumerate_trace_from_dfa_solver(solver, automaton,varmap,num_solutions)
    del solver
    return words

if __name__=="__main__":
    automaton = create_automaton_from_yaml("config.yaml")
    words = generate_traces(10,automaton,7)
    print_aq_sequences(words)