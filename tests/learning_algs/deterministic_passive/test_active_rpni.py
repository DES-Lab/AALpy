import random
import unittest

from aalpy.automata import Dfa, DfaState, MooreMachine, MooreState
from aalpy.learning_algs.deterministic_passive.active_RPNI import (
    RandomWordSampler, RpniActiveSampler, run_active_RPNI,
)
from aalpy.SULs import AutomatonSUL
from aalpy.utils.ModelChecking import bisimilar


def even_a_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


def three_state_moore():
    q0 = MooreState('q0', 0)
    q1 = MooreState('q1', 1)
    q2 = MooreState('q2', 2)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q2, 'b': q0}
    q2.transitions = {'a': q2, 'b': q1}
    return MooreMachine(q0, [q0, q1, q2])


class ConstantSampler(RpniActiveSampler):
    """Deterministic sampler used to exercise the RpniActiveSampler contract with fixed words."""

    def __init__(self, words):
        self.words = words

    def sample(self, sul, model):
        samples = []
        for word in self.words:
            outputs = sul.query(word)
            samples.append((word, outputs[-1]))
        return samples


class NoOpSampler(RpniActiveSampler):
    """Sampler that never queries the SUL, used where only the initial data matters."""

    def sample(self, sul, model):
        return []


def bootstrap_data(ground_truth, sul, initial_output):
    """
    Builds a minimal initial data set that includes one transition per alphabet symbol, so that the very
    first hypothesis learned by run_active_RPNI already has an input alphabet for RandomWordSampler to
    sample from (it derives its alphabet from the transitions of the current hypothesis).
    """
    data = [((), initial_output)]
    for letter in ground_truth.get_input_alphabet():
        outputs = sul.query((letter,))
        data.append(((letter,), outputs[-1]))
    return data


class TestRunActiveRpni(unittest.TestCase):
    def test_learns_correct_dfa_with_random_word_sampler(self):
        random.seed(1)
        ground_truth = even_a_dfa()
        sul = AutomatonSUL(ground_truth)
        sampler = RandomWordSampler(num_walks=20, min_walk_len=1, max_walk_len=5)
        data = bootstrap_data(ground_truth, sul, True)

        learned = run_active_RPNI(data=data, sul=sul, sampler=sampler, n_iter=5,
                                   automaton_type='dfa', print_info=False)

        self.assertTrue(bisimilar(learned, ground_truth))

    def test_learns_correct_moore_machine_with_enough_iterations(self):
        random.seed(2)
        ground_truth = three_state_moore()
        sul = AutomatonSUL(ground_truth)
        sampler = RandomWordSampler(num_walks=30, min_walk_len=1, max_walk_len=6)
        data = bootstrap_data(ground_truth, sul, 0)

        learned = run_active_RPNI(data=data, sul=sul, sampler=sampler, n_iter=6,
                                   automaton_type='moore', print_info=False)

        self.assertTrue(bisimilar(learned, ground_truth))

    def test_data_grows_with_each_iteration(self):
        ground_truth = even_a_dfa()
        sul = AutomatonSUL(ground_truth)
        sampler = ConstantSampler([('a',), ('b', 'a')])
        data = [((), True)]

        run_active_RPNI(data=data, sul=sul, sampler=sampler, n_iter=3,
                         automaton_type='dfa', print_info=False)

        # 3 iterations each adding 2 fixed samples on top of the initial one
        self.assertEqual(len(data), 1 + 3 * 2)

    def test_returns_none_when_data_is_inconsistent(self):
        ground_truth = even_a_dfa()
        sul = AutomatonSUL(ground_truth)
        # conflicting labels for the same (empty) sequence makes the data inconsistent from the start
        data = [((), True), ((), False)]

        learned = run_active_RPNI(data=data, sul=sul, sampler=NoOpSampler(), n_iter=2,
                                   automaton_type='dfa', print_info=False)
        self.assertIsNone(learned)


if __name__ == '__main__':
    unittest.main()
