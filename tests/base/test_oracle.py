import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Dfa, DfaState
from aalpy.base import Oracle


def parity_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


class DummyOracle(Oracle):
    """Minimal concrete Oracle used to exercise the shared base behaviour."""

    def find_cex(self, hypothesis):
        return None


class TestOracle(unittest.TestCase):
    def test_constructor_sets_alphabet_and_sul(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        oracle = DummyOracle(['a', 'b'], sul)
        self.assertEqual(oracle.alphabet, ['a', 'b'])
        self.assertIs(oracle.sul, sul)
        self.assertEqual(oracle.num_queries, 0)
        self.assertEqual(oracle.num_steps, 0)

    def test_find_cex_is_abstract_and_must_be_implemented(self):
        with self.assertRaises(TypeError):
            Oracle(['a'], AutomatonSUL(parity_dfa()))

    def test_reset_hyp_and_sul_resets_hypothesis_to_initial(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        oracle = DummyOracle(dfa.get_input_alphabet(), sul)

        dfa.step('a')
        self.assertIsNot(dfa.current_state, dfa.initial_state)

        oracle.reset_hyp_and_sul(dfa)
        self.assertIs(dfa.current_state, dfa.initial_state)

    def test_reset_hyp_and_sul_resets_sul_automaton(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        oracle = DummyOracle(dfa.get_input_alphabet(), sul)

        sul.step('a')
        self.assertIsNot(sul.automaton.current_state, sul.automaton.initial_state)

        oracle.reset_hyp_and_sul(dfa)
        self.assertIs(sul.automaton.current_state, sul.automaton.initial_state)

    def test_reset_hyp_and_sul_increments_num_queries(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        oracle = DummyOracle(dfa.get_input_alphabet(), sul)

        oracle.reset_hyp_and_sul(dfa)
        oracle.reset_hyp_and_sul(dfa)
        self.assertEqual(oracle.num_queries, 2)


if __name__ == '__main__':
    unittest.main()
