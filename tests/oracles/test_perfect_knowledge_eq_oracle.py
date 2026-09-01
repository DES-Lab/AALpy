import unittest

from aalpy.automata import Dfa, DfaState
from aalpy.oracles import PerfectKnowledgeEqOracle
from aalpy.SULs import AutomatonSUL


def parity_dfa():
    """2-state complete, minimal DFA accepting words with an even number of 'a's."""
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    dfa = Dfa(q0, [q0, q1])
    dfa.compute_prefixes()
    return dfa


def parity_dfa_relabeled():
    """Behaviorally identical to parity_dfa, but built from differently-named/ordered states."""
    r1 = DfaState('r1', is_accepting=False)
    r0 = DfaState('r0', is_accepting=True)
    r0.transitions = {'a': r1, 'b': r0}
    r1.transitions = {'a': r0, 'b': r1}
    dfa = Dfa(r0, [r1, r0])
    dfa.compute_prefixes()
    return dfa


def parity_dfa_with_wrong_transition():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q1, 'b': q1}  # 'a' from q1 should go back to q0, not self-loop
    dfa = Dfa(q0, [q0, q1])
    dfa.compute_prefixes()
    return dfa


class PerfectKnowledgeEqOracleTests(unittest.TestCase):

    def test_finds_cex_for_wrong_transition(self):
        ground_truth = parity_dfa()
        hypothesis = parity_dfa_with_wrong_transition()

        oracle = PerfectKnowledgeEqOracle(['a', 'b'], AutomatonSUL(ground_truth), ground_truth)
        cex = oracle.find_cex(hypothesis)

        self.assertIsNotNone(cex)
        ground_truth.reset_to_initial()
        hypothesis.reset_to_initial()
        sul_out = [ground_truth.step(i) for i in cex]
        hyp_out = [hypothesis.step(i) for i in cex]
        self.assertNotEqual(sul_out[-1], hyp_out[-1])

    def test_no_cex_for_behaviorally_equivalent_but_structurally_different_hypothesis(self):
        ground_truth = parity_dfa()
        hypothesis = parity_dfa_relabeled()

        oracle = PerfectKnowledgeEqOracle(['a', 'b'], AutomatonSUL(ground_truth), ground_truth)
        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_deterministically_finds_cex_on_first_try_no_randomness_involved(self):
        # unlike the other oracles, this one has direct access to ground truth, so it must succeed every time
        for _ in range(5):
            ground_truth = parity_dfa()
            hypothesis = parity_dfa_with_wrong_transition()

            oracle = PerfectKnowledgeEqOracle(['a', 'b'], AutomatonSUL(ground_truth), ground_truth)
            self.assertIsNotNone(oracle.find_cex(hypothesis))

    def test_missing_state_in_hypothesis_is_detected(self):
        ground_truth = parity_dfa()

        single_state_hyp = DfaState('only', is_accepting=True)
        single_state_hyp.transitions = {'a': single_state_hyp, 'b': single_state_hyp}
        hypothesis = Dfa(single_state_hyp, [single_state_hyp])
        hypothesis.compute_prefixes()

        oracle = PerfectKnowledgeEqOracle(['a', 'b'], AutomatonSUL(ground_truth), ground_truth)
        cex = oracle.find_cex(hypothesis)

        self.assertIsNotNone(cex)
        ground_truth.reset_to_initial()
        hypothesis.reset_to_initial()
        sul_out = [ground_truth.step(i) for i in cex]
        hyp_out = [hypothesis.step(i) for i in cex]
        self.assertNotEqual(sul_out[-1], hyp_out[-1])


if __name__ == '__main__':
    unittest.main()
