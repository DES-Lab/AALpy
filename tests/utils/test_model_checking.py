import unittest

from aalpy.automata import Dfa, DfaState, MooreMachine, MooreState, MealyMachine, MealyState, Onfsm, OnfsmState
from aalpy.utils.BenchmarkSULs import get_Angluin_dfa
from aalpy.utils.ModelChecking import bisimilar, compare_automata


def parity_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


def non_bisimilar_dfa():
    # accepts everything (always accepting), differs from parity_dfa on 'a'
    q0 = DfaState('q0', is_accepting=True)
    q0.transitions = {'a': q0, 'b': q0}
    return Dfa(q0, [q0])


def renamed_parity_dfa():
    r0 = DfaState('r0', is_accepting=True)
    r1 = DfaState('r1', is_accepting=False)
    r0.transitions = {'a': r1, 'b': r0}
    r1.transitions = {'a': r0, 'b': r1}
    return Dfa(r0, [r0, r1])


def parity_moore():
    q0 = MooreState('q0', output=True)
    q1 = MooreState('q1', output=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return MooreMachine(q0, [q0, q1])


def parity_mealy():
    q0 = MealyState('q0')
    q1 = MealyState('q1')
    q0.transitions = {'a': q1, 'b': q0}
    q0.output_fun = {'a': 'x', 'b': 'y'}
    q1.transitions = {'a': q0, 'b': q1}
    q1.output_fun = {'a': 'y', 'b': 'x'}
    return MealyMachine(q0, [q0, q1])


class TestBisimilar(unittest.TestCase):
    def test_identical_automaton_is_bisimilar_to_itself_copy(self):
        dfa = parity_dfa()
        self.assertTrue(bisimilar(dfa, dfa))

    def test_isomorphic_automata_with_different_state_ids_are_bisimilar(self):
        dfa1 = parity_dfa()
        dfa2 = renamed_parity_dfa()
        self.assertTrue(bisimilar(dfa1, dfa2))

    def test_non_bisimilar_dfa_returns_false(self):
        dfa1 = parity_dfa()
        dfa2 = non_bisimilar_dfa()
        self.assertFalse(bisimilar(dfa1, dfa2))

    def test_non_bisimilar_dfa_counterexample(self):
        dfa1 = parity_dfa()
        dfa2 = non_bisimilar_dfa()
        cex = bisimilar(dfa1, dfa2, return_cex=True)
        self.assertIsNotNone(cex)
        dfa1.reset_to_initial()
        dfa2.reset_to_initial()
        out1 = [dfa1.step(i) for i in cex]
        out2 = [dfa2.step(i) for i in cex]
        self.assertNotEqual(out1[-1], out2[-1])

    def test_bisimilar_automata_return_cex_gives_none(self):
        dfa1 = parity_dfa()
        dfa2 = renamed_parity_dfa()
        self.assertIsNone(bisimilar(dfa1, dfa2, return_cex=True))

    def test_moore_bisimilarity(self):
        moore1 = parity_moore()
        moore2 = parity_moore()
        self.assertTrue(bisimilar(moore1, moore2))

    def test_mealy_bisimilarity(self):
        mealy1 = parity_mealy()
        mealy2 = parity_mealy()
        self.assertTrue(bisimilar(mealy1, mealy2))

    def test_mealy_output_mismatch_detected(self):
        mealy1 = parity_mealy()
        mealy2 = parity_mealy()
        mealy2.initial_state.output_fun['a'] = 'different'
        self.assertFalse(bisimilar(mealy1, mealy2))

    def test_different_automaton_types_raises(self):
        dfa = parity_dfa()
        moore = parity_moore()
        with self.assertRaises(ValueError):
            bisimilar(dfa, moore)

    def test_unsupported_automaton_type_raises(self):
        onfsm = Onfsm(OnfsmState('q0'), [OnfsmState('q0')])
        with self.assertRaises(NotImplementedError):
            bisimilar(onfsm, onfsm)

    def test_different_enabled_inputs_not_bisimilar(self):
        q0 = DfaState('q0', is_accepting=True)
        q0.transitions = {'a': q0}
        dfa1 = Dfa(q0, [q0])

        r0 = DfaState('r0', is_accepting=True)
        r0.transitions = {'a': r0, 'b': r0}
        dfa2 = Dfa(r0, [r0])

        self.assertFalse(bisimilar(dfa1, dfa2))


class TestCompareAutomata(unittest.TestCase):
    def test_identical_automata_no_counterexamples(self):
        dfa1 = get_Angluin_dfa()
        dfa2 = get_Angluin_dfa()
        cexs = compare_automata(dfa1, dfa2)
        self.assertEqual(cexs, [])

    def test_same_object_no_counterexamples(self):
        dfa = get_Angluin_dfa()
        cexs = compare_automata(dfa, dfa)
        self.assertEqual(cexs, [])

    def test_different_alphabets_raises(self):
        dfa1 = get_Angluin_dfa()
        q0 = DfaState('q0', is_accepting=True)
        q0.transitions = {'x': q0, 'y': q0}
        dfa2 = Dfa(q0, [q0])
        with self.assertRaises(AssertionError):
            compare_automata(dfa1, dfa2)

    def test_finds_counterexample_for_differing_automata(self):
        dfa1 = parity_dfa()
        dfa2 = non_bisimilar_dfa()
        cexs = compare_automata(dfa1, dfa2, num_cex=5)
        self.assertGreater(len(cexs), 0)
        for cex in cexs:
            dfa1.reset_to_initial()
            dfa2.reset_to_initial()
            out1 = [dfa1.step(i) for i in cex][-1]
            out2 = [dfa2.step(i) for i in cex][-1]
            self.assertNotEqual(out1, out2)

    def test_counterexamples_sorted_by_length(self):
        dfa1 = parity_dfa()
        dfa2 = non_bisimilar_dfa()
        cexs = compare_automata(dfa1, dfa2, num_cex=8)
        lengths = [len(c) for c in cexs]
        self.assertEqual(lengths, sorted(lengths))


if __name__ == '__main__':
    unittest.main()
