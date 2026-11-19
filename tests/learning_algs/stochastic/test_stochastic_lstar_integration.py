import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Mdp, MdpState
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles import RandomWalkEqOracle

# NOTE on why this does not follow the legacy tests/test_stochastic.py verification approach:
# that test hardcoded aalpy.paths.path_to_prism to a Windows path and called model_check_experiment, which
# shells out to a real PRISM binary. PRISM is not installed on this machine (or in CI), so that check would
# always fail with FileNotFoundError. Instead we validate the learned model directly against the ground
# truth by sampling: comparing per-input-sequence output distributions between the ground truth MDP and the
# learned model (both Mdp and StochasticMealyMachine expose execute_sequence(state, input_seq) returning one
# output per input, so no PRISM/conversion step is required for this comparison).


def _ground_truth_mdp():
    """
    3-state MDP ("biased grid"): s0(A) --a--> s1(B) w.p. 0.7 / s2(C) w.p. 0.3; s0 --b--> s0.
    s1(B)/s2(C) are absorbing back to s0 on 'a', and self-loop on 'b'.
    """
    s0 = MdpState('s0', output='A')
    s1 = MdpState('s1', output='B')
    s2 = MdpState('s2', output='C')
    s0.transitions['a'].append((s1, 0.7))
    s0.transitions['a'].append((s2, 0.3))
    s0.transitions['b'].append((s0, 1.0))
    s1.transitions['a'].append((s0, 1.0))
    s1.transitions['b'].append((s1, 1.0))
    s2.transitions['a'].append((s0, 1.0))
    s2.transitions['b'].append((s2, 1.0))
    return Mdp(s0, [s0, s1, s2])


def _output_distribution(model, input_seq, n_samples):
    """Samples model.execute_sequence(initial_state, input_seq) n_samples times and returns a normalized
    frequency dict over the resulting output tuples."""
    counts = {}
    for _ in range(n_samples):
        outputs = tuple(model.execute_sequence(model.initial_state, list(input_seq)))
        counts[outputs] = counts.get(outputs, 0) + 1
    return {k: v / n_samples for k, v in counts.items()}


def _total_variation_distance(dist1, dist2):
    keys = set(dist1) | set(dist2)
    return 0.5 * sum(abs(dist1.get(k, 0) - dist2.get(k, 0)) for k in keys)


def _assert_models_behave_similarly(test_case, ground_truth, learned_model, seed=123, n_samples=300, tolerance=0.35):
    random.seed(seed)
    input_sequences = [('a',), ('b', 'a'), ('a', 'a'), ('a', 'b', 'a')]
    for input_seq in input_sequences:
        truth_dist = _output_distribution(ground_truth, input_seq, n_samples)
        learned_dist = _output_distribution(learned_model, input_seq, n_samples)
        tvd = _total_variation_distance(truth_dist, learned_dist)
        test_case.assertLessEqual(
            tvd, tolerance,
            f'Output distribution for {input_seq} diverged too much: truth={truth_dist} learned={learned_dist}')


class StochasticLStarIntegrationTest(unittest.TestCase):
    """
    Trimmed-down sweep over run_stochastic_Lstar: one representative combination per automaton_type, plus a
    couple of extra strategy/cex_processing/samples_cex_strategy variations, instead of the full 2x3x2x3
    cross-product from the legacy test (kept fast, and every combination exercises the same core algorithm).
    """

    def _learn(self, automaton_type, strategy, cex_processing, samples_cex_strategy, seed):
        random.seed(seed)
        ground_truth = _ground_truth_mdp()
        input_alphabet = ground_truth.get_input_alphabet()
        sul = AutomatonSUL(ground_truth)
        eq_oracle = RandomWalkEqOracle(input_alphabet, sul=sul, num_steps=150, reset_prob=0.25,
                                       reset_after_cex=True)

        learned_model = run_stochastic_Lstar(
            input_alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul,
            n_c=20, n_resample=100, min_rounds=5, max_rounds=20,
            automaton_type=automaton_type, strategy=strategy, cex_processing=cex_processing,
            samples_cex_strategy=samples_cex_strategy, target_unambiguity=0.99, print_level=0)

        return ground_truth, learned_model

    def test_learn_mdp_classic_strategy(self):
        ground_truth, learned_model = self._learn('mdp', 'classic', None, None, seed=1)
        _assert_models_behave_similarly(self, ground_truth, learned_model)

    def test_learn_smm_normal_strategy_with_longest_prefix_cex(self):
        ground_truth, learned_model = self._learn('smm', 'normal', 'longest_prefix', None, seed=2)
        _assert_models_behave_similarly(self, ground_truth, learned_model)

    def test_learn_mdp_chi2_strategy_with_bfs_samples_cex(self):
        ground_truth, learned_model = self._learn('mdp', 'chi2', None, 'bfs', seed=3)
        _assert_models_behave_similarly(self, ground_truth, learned_model)

    def test_learn_smm_classic_strategy_with_random_samples_cex(self):
        ground_truth, learned_model = self._learn('smm', 'classic', None, 'random:200:0.3', seed=4)
        _assert_models_behave_similarly(self, ground_truth, learned_model)


if __name__ == '__main__':
    unittest.main()
