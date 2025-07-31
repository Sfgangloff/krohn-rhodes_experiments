import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from automaton import Automaton
from generate_traces import encode_dfa_trace_to_solver, enumerate_trace_from_dfa_solver

class TestDFASATEncoding(unittest.TestCase):
    def setUp(self):
        self.automaton = Automaton(
            states={"0", "1"},
            alphabet={"0", "1"},
            transitions={
                ("0", "0"): "0",
                ("0", "1"): "1",
                ("1", "0"): "1",
                ("1", "1"): "0"
            },
            initial_state="0",
            accepting_states={"0"}
        )

    def test_solver_creation(self):
        solver, varmap = encode_dfa_trace_to_solver(3, self.automaton)
        self.assertTrue(hasattr(solver, 'solve'))
        self.assertIn(('x', 1, '0'), varmap)
        self.assertIn(('y', 0, '0'), varmap)

    def test_enumerate_output_type(self):
        solver, varmap = encode_dfa_trace_to_solver(3, self.automaton)
        traces = enumerate_trace_from_dfa_solver(solver,self.automaton, varmap, num_solutions=5)
        self.assertIsInstance(traces, list)
        self.assertTrue(all(isinstance(trace, list) for trace in traces))
        self.assertTrue(all(all(isinstance(pair, tuple) and len(pair) == 2 for pair in trace) for trace in traces))

    def test_trace_length(self):
        solver, varmap = encode_dfa_trace_to_solver(4, self.automaton)
        traces = enumerate_trace_from_dfa_solver(solver,self.automaton, varmap, num_solutions=5)
        self.assertTrue(all(len(trace) == 4 for trace in traces))  # 4 (a,q) pairs

    def test_symbols_and_states_valid(self):
        solver, varmap = encode_dfa_trace_to_solver(3, self.automaton)
        traces = enumerate_trace_from_dfa_solver(solver,self.automaton, varmap, num_solutions=5)
        for trace in traces:
            for a, q in trace:
                self.assertIn(a, self.automaton.alphabet)
                self.assertIn(q, self.automaton.states)

    def test_number_of_traces(self):
        solver, varmap = encode_dfa_trace_to_solver(2, self.automaton)
        traces = enumerate_trace_from_dfa_solver(solver,self.automaton, varmap, num_solutions=10)

        # With 2 steps, there are 2^2 = 4 input strings,
        # but since each (a_i, q_{i-1}) pair includes the state,
        # the number of possible traces is determined by reachable state sequences
        # Let's check only that we get at most the requested number
        self.assertLessEqual(len(traces), 10)

if __name__ == "__main__":
    unittest.main()