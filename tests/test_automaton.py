import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automaton import Automaton,cascade_automata

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
        self.assertEqual(cascade.transition(('q0', 'p0'), 'a'), ('q1', 'p1'))
        # From ('q1','p1'), input 'b' → q1, output_map(q1) = x → p0
        self.assertEqual(cascade.transition(('q1', 'p1'), 'b'), ('q1', 'p0'))

    def test_is_accepting(self):
        cascade = cascade_automata(self.A1, self.A2, self.output_map)
        self.assertTrue(cascade.is_accepting(('q1', 'p1')))
        self.assertFalse(cascade.is_accepting(('q1', 'p0')))

if __name__ == "__main__":
    unittest.main()