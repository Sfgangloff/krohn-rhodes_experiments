import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automaton import Automaton,cascade_automata,cascade_multiple_automata,flatten_dfa_states
from utils import flatten

class TestAutomaton(unittest.TestCase):
    def setUp(self):
        self.states = {'q0', 'q1'}
        self.alphabet = {'a', 'b'}
        self.transitions = {
            ('q0', 'a'): 'q1',
            ('q0', 'b'): 'q0',
            ('q1', 'a'): 'q0',
            ('q1', 'b'): 'q1'
        }
        self.initial_state = 'q0'
        self.accepting_states = {'q0'}

        self.automaton = Automaton(
            states=self.states,
            alphabet=self.alphabet,
            transitions=self.transitions,
            initial_state=self.initial_state,
            accepting_states=self.accepting_states
        )

    def test_transition_valid(self):
        self.assertEqual(self.automaton.transition('q0', 'a'), 'q1')
        self.assertEqual(self.automaton.transition('q1', 'b'), 'q1')
        self.assertEqual(self.automaton.transition('q0', 'b'), 'q0')

    def test_transition_invalid(self):
        self.assertIsNone(self.automaton.transition('q0', 'c'))  # symbol not in alphabet
        self.assertIsNone(self.automaton.transition('q2', 'a'))  # state not in DFA

    def test_is_accepting(self):
        self.assertTrue(self.automaton.is_accepting('q0'))
        self.assertFalse(self.automaton.is_accepting('q1'))
        self.assertFalse(self.automaton.is_accepting('q2'))  # state not in DFA

    def test_initial_state(self):
        self.assertEqual(self.automaton.initial_state, 'q0')

    def test_states_and_alphabet(self):
        self.assertSetEqual(self.automaton.states, {'q0', 'q1'})
        self.assertSetEqual(self.automaton.alphabet, {'a', 'b'})

class TestCascadeAutomata(unittest.TestCase):
    def setUp(self):
        # Automaton A1: transitions on {0,1}, outputs state q0 or q1
        self.A1 = Automaton(
            states={'q0', 'q1'},
            alphabet={'a', 'b'},
            transitions={
                ('q0', 'a'): 'q1',
                ('q0', 'b'): 'q0',
                ('q1', 'a'): 'q0',
                ('q1', 'b'): 'q1'
            },
            initial_state='q0',
            accepting_states={'q1'}
        )

        # Automaton A2: input alphabet is 'x', 'y'
        self.A2 = Automaton(
            states={'p0', 'p1'},
            alphabet={'x', 'y'},
            transitions={
                ('p0', 'x'): 'p1',
                ('p0', 'y'): 'p0',
                ('p1', 'x'): 'p0',
                ('p1', 'y'): 'p1'
            },
            initial_state='p0',
            accepting_states={'p1'}
        )

        # Define output_map: maps A1 state to A2 input
        self.output_map = lambda q: 'x' if q == 'q1' else 'y'

    def test_states_and_alphabet(self):
        cascade = cascade_automata(self.A1, self.A2, self.output_map)
        expected_states = {('q0', 'p0'), ('q0', 'p1'), ('q1', 'p0'), ('q1', 'p1')}
        self.assertEqual(cascade.states, expected_states)
        self.assertEqual(cascade.alphabet, self.A1.alphabet)

    def test_initial_state(self):
        cascade = cascade_automata(self.A1, self.A2, self.output_map)
        self.assertEqual(cascade.initial_state, ('q0', 'p0'))

    def test_accepting_states(self):
        cascade = cascade_automata(self.A1, self.A2, self.output_map)
        expected_accepting = {('q0', 'p1'), ('q1', 'p1')}
        self.assertEqual(cascade.accepting_states, expected_accepting)

    def test_transitions(self):
        cascade = cascade_automata(self.A1, self.A2, self.output_map)
        # Check a few expected transitions manually
        # From ('q0','p0'), input 'a' → q1, output_map(q1) = x → p1
        self.assertEqual(cascade.transition(('q0', 'p0'), 'a'), ('q1', 'p0'))
        # From ('q1','p1'), input 'b' → q1, output_map(q1) = x → p0
        self.assertEqual(cascade.transition(('q1', 'p1'), 'b'), ('q1', 'p0'))

    def test_is_accepting(self):
        cascade = cascade_automata(self.A1, self.A2, self.output_map)
        self.assertTrue(cascade.is_accepting(('q1', 'p1')))
        self.assertFalse(cascade.is_accepting(('q1', 'p0')))

class TestCascadeMultipleAutomata(unittest.TestCase):
    def setUp(self):
        # A1 and A2 as before
        self.A1 = Automaton(
            states={'q0', 'q1'},
            alphabet={'a', 'b'},
            transitions={
                ('q0', 'a'): 'q1',
                ('q0', 'b'): 'q0',
                ('q1', 'a'): 'q0',
                ('q1', 'b'): 'q1'
            },
            initial_state='q0',
            accepting_states={'q1'}
        )

        self.A2 = Automaton(
            states={'p0', 'p1'},
            alphabet={'x', 'y'},
            transitions={
                ('p0', 'x'): 'p1',
                ('p0', 'y'): 'p0',
                ('p1', 'x'): 'p0',
                ('p1', 'y'): 'p1'
            },
            initial_state='p0',
            accepting_states={'p1'}
        )

        self.A3 = Automaton(
            states={'r0', 'r1'},
            alphabet={'m', 'n'},
            transitions={
                ('r0', 'm'): 'r1',
                ('r0', 'n'): 'r0',
                ('r1', 'm'): 'r0',
                ('r1', 'n'): 'r1'
            },
            initial_state='r0',
            accepting_states={'r1'}
        )

        # output maps
        self.output_map_1 = lambda q: 'x' if q == 'q1' else 'y'  # q ∈ A1.states → A2 input
        self.output_map_2 = lambda p: 'm' if p[1] == 'p1' else 'n'  # p ∈ A2.states → A3 input

    def test_cascade_many_states(self):
        cascade = cascade_multiple_automata([self.A1, self.A2, self.A3], [self.output_map_1, self.output_map_2])
        expected_states = {((q, p), r) for q in self.A1.states for p in self.A2.states for r in self.A3.states}
        self.assertEqual(cascade.states, expected_states)

    def test_cascade_many_initial_state(self):
        cascade = cascade_multiple_automata([self.A1, self.A2, self.A3], [self.output_map_1, self.output_map_2])
        self.assertEqual(cascade.initial_state, (('q0', 'p0'), 'r0'))

    def test_cascade_many_accepting_states(self):
        cascade = cascade_multiple_automata([self.A1, self.A2, self.A3], [self.output_map_1, self.output_map_2])
        expected_accepting = {
            ((q, p), r)
            for q in self.A1.states
            for p in self.A2.states
            for r in self.A3.states
            if self.A3.is_accepting(r)
        }
        self.assertEqual(cascade.accepting_states, expected_accepting)

    def test_cascade_many_transitions(self):
        cascade = cascade_multiple_automata([self.A1, self.A2, self.A3], [self.output_map_1, self.output_map_2])
        # Step-by-step:
        # ('q0', 'p0', 'r0'), input 'a' →
        # q1 = A1(q0, 'a'), output_map_1(q1) = 'x'
        # p1 = A2(p0, 'x'), output_map_2(p1) = 'm'
        # r1 = A3(r0, 'm')
        self.assertEqual(cascade.transition((('q0', 'p0'), 'r0'), 'a'), (('q1', 'p0'), 'r0'))

        # Another transition
        # ('q1', 'p1', 'r1'), input 'b'
        # q1 = A1(q1, 'b') = q1, output_map_1(q1) = x
        # p0 = A2(p1, 'x') = p0, output_map_2(p0) = n
        # r1 = A3(r1, 'n') = r1
        self.assertEqual(cascade.transition((('q1', 'p1'), 'r1'), 'b'), (('q1', 'p0'), 'r0'))

    def test_cascade_many_accepting_behavior(self):
        cascade = cascade_multiple_automata([self.A1, self.A2, self.A3], [self.output_map_1, self.output_map_2])
        self.assertTrue(cascade.is_accepting((('q1', 'p1'), 'r1')))
        self.assertFalse(cascade.is_accepting((('q1', 'p1'), 'r0')))

class TestFlattenFunctions(unittest.TestCase):
    def test_flatten_simple(self):
        self.assertEqual(flatten('q0'), ('q0',))
        self.assertEqual(flatten(('q0', 'x')), ('q0', 'x'))
        self.assertEqual(flatten((('q0', 'x'), 'q1')), ('q0', 'x', 'q1'))
        self.assertEqual(flatten(((('a', 'b'), 'c'), 'd')), ('a', 'b', 'c', 'd'))

    def test_flatten_empty_tuple(self):
        self.assertEqual(flatten(()), ())

    def test_flatten_nested_mixed(self):
        nested = ((('q0', 'a'), ('q1',)), 'q2')
        self.assertEqual(flatten(nested), ('q0', 'a', 'q1', 'q2'))


class TestFlattenDFA(unittest.TestCase):
    def setUp(self):
        # DFA with nested tuple states
        self.dfa = Automaton(
            states={
                (('q0', 'x'), 'r0'),
                (('q0', 'x'), 'r1'),
                (('q1', 'y'), 'r0'),
                (('q1', 'y'), 'r1')
            },
            alphabet={'a', 'b'},
            transitions={
                ((('q0', 'x'), 'r0'), 'a'): ((('q1', 'y'), 'r0')),
                ((('q0', 'x'), 'r1'), 'a'): ((('q1', 'y'), 'r1')),
                ((('q1', 'y'), 'r0'), 'b'): ((('q0', 'x'), 'r0')),
                ((('q1', 'y'), 'r1'), 'b'): ((('q0', 'x'), 'r1')),
            },
            initial_state=(('q0', 'x'), 'r0'),
            accepting_states={(('q1', 'y'), 'r1')}
        )

    def test_flatten_dfa_states_structure(self):
        flat_dfa = flatten_dfa_states(self.dfa)
        # States are flattened
        expected_states = {
            ('q0', 'x', 'r0'),
            ('q0', 'x', 'r1'),
            ('q1', 'y', 'r0'),
            ('q1', 'y', 'r1'),
        }
        self.assertEqual(flat_dfa.states, expected_states)

        # Initial state
        self.assertEqual(flat_dfa.initial_state, ('q0', 'x', 'r0'))

        # Accepting states
        self.assertEqual(flat_dfa.accepting_states, {('q1', 'y', 'r1')})

    def test_flatten_dfa_transitions(self):
        flat_dfa = flatten_dfa_states(self.dfa)
        # Check one transition: from ('q0','x','r0') via 'a' → ('q1','y','r0')
        self.assertEqual(
            flat_dfa.transition(('q0', 'x', 'r0'), 'a'),
            ('q1', 'y', 'r0')
        )

        self.assertEqual(
            flat_dfa.transition(('q1', 'y', 'r1'), 'b'),
            ('q0', 'x', 'r1')
        )

if __name__ == "__main__":
    unittest.main()
    # tests = TestCascadeMultipleAutomata()
    # tests.setUp()
    # tests.test_cascade_many_transitions()