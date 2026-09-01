import random

import pytest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_non_det_Lstar
from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
from aalpy.utils import get_benchmark_ONFSM

pytestmark = pytest.mark.exhaustive

# Full sweep this repo used to run at the root (tests/test_non_deterministic.py) before it was trimmed
# down to 8 seeded iterations (see the sibling test_onfsm_lstar.py): 100 fresh iterations with a wider
# random-walk/sampling budget. ONFSM learning is inherently randomized, so more iterations catch rarer
# non-determinism/sampling edge cases that a small seeded run can miss.
ITERATIONS = 100


def test_learns_benchmark_onfsm_with_correct_state_count_and_no_cex_exhaustive():
    onfsm = get_benchmark_ONFSM()
    alphabet = onfsm.get_input_alphabet()

    for i in range(ITERATIONS):
        sul = AutomatonSUL(onfsm)

        oracle = RandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=2, max_walk_len=5)

        learned_onfsm = run_non_det_Lstar(alphabet, sul, oracle, n_sampling=50, print_level=0)

        eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=10000, reset_prob=0.09,
                                        reset_after_cex=True)

        cex = eq_oracle.find_cex(learned_onfsm)

        assert cex is None, f'iteration {i}: independent oracle found a counterexample'
        assert len(learned_onfsm.states) == len(onfsm.states), \
            f'iteration {i}: learned model has wrong number of states'
