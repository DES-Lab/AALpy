import random
import unittest

import pytest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles import RandomWalkEqOracle
from tests.learning_algs.stochastic.test_stochastic_lstar_integration import (
    _assert_models_behave_similarly, _ground_truth_mdp)

pytestmark = pytest.mark.exhaustive

# Full cross-product this repo's legacy tests/test_stochastic.py swept (minus its PRISM dependency, dropped
# for the same reason as the sibling test_stochastic_lstar_integration.py): 2 automaton_types x 3 strategies
# x 2 cex_processing x 3 samples_cex_strategy = 36 combinations, instead of the trimmed handful kept in the
# fast default suite.
AUTOMATON_TYPES = ['mdp', 'smm']
STRATEGIES = ['classic', 'normal', 'chi2']
CEX_PROCESSING = [None, 'longest_prefix']
SAMPLES_CEX_STRATEGY = [None, 'bfs', 'random:200:0.3']

TEST_CASES = [
    (automaton_type, strategy, cex_processing, samples_cex_strategy)
    for automaton_type in AUTOMATON_TYPES
    for strategy in STRATEGIES
    for cex_processing in CEX_PROCESSING
    for samples_cex_strategy in SAMPLES_CEX_STRATEGY
]


class StochasticLStarExhaustiveTest(unittest.TestCase):
    def _learn(self, automaton_type, strategy, cex_processing, samples_cex_strategy, seed):
        random.seed(seed)
        ground_truth = _ground_truth_mdp()
        input_alphabet = ground_truth.get_input_alphabet()
        sul = AutomatonSUL(ground_truth)
        eq_oracle = RandomWalkEqOracle(input_alphabet, sul=sul, num_steps=200, reset_prob=0.25,
                                       reset_after_cex=True)

        learned_model = run_stochastic_Lstar(
            input_alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul,
            n_c=20, n_resample=1000, min_rounds=10, max_rounds=100,
            automaton_type=automaton_type, strategy=strategy, cex_processing=cex_processing,
            samples_cex_strategy=samples_cex_strategy, target_unambiguity=0.99, print_level=0)

        return ground_truth, learned_model

    def test_full_sweep(self):
        for seed, (automaton_type, strategy, cex_processing, samples_cex_strategy) in enumerate(TEST_CASES):
            with self.subTest(automaton_type=automaton_type, strategy=strategy,
                              cex_processing=cex_processing, samples_cex_strategy=samples_cex_strategy):
                ground_truth, learned_model = self._learn(automaton_type, strategy, cex_processing,
                                                           samples_cex_strategy, seed=seed)
                _assert_models_behave_similarly(self, ground_truth, learned_model)


if __name__ == '__main__':
    unittest.main()
