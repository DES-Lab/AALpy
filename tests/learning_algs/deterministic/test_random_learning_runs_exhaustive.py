import random

import pytest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_KV, run_Lsharp, run_Lstar
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.utils import generate_random_deterministic_automata
from aalpy.utils.ModelChecking import bisimilar

pytestmark = pytest.mark.exhaustive

# Full sweep this repo used to run at the root (tests/test_deterministic_learning_runs.py) before it was
# trimmed down to a fast default (see the sibling test_random_learning_runs.py): 50 seeds x 16 sizes x 3
# automaton types x 3 algorithms = 7200 cases. Wide sweeps like this are what caught e.g. the
# linear_cex_processing direction bug and the ObservationTree off-by-one during this test suite's build-out
# - a handful of seeds can get lucky and never hit the input sequence that triggers a given bug.
SEEDS = list(range(50))
MODEL_SIZES = [
    (2, 2, 2),
    (2, 2, 3),
    (3, 2, 2),
    (3, 2, 3),
    (3, 3, 2),
    (4, 2, 3),
    (4, 3, 2),
    (5, 3, 3),
    (6, 2, 3),
    (6, 3, 2),
    (10, 2, 3),
    (10, 2, 4),
    (10, 2, 2),
    (10, 2, 3),
    (20, 5, 5),
    (30, 3, 4),
]

TEST_CASES = [
    pytest.param(
        learning_alg,
        automaton_type,
        seed_val,
        num_states,
        input_size,
        output_size,
        id=f"{learning_alg.__name__}-{automaton_type}-states={num_states}-seed={seed_val}",
    )
    for num_states, input_size, output_size in MODEL_SIZES
    for seed_val in SEEDS
    for automaton_type in ['dfa', 'moore', 'mealy']
    for learning_alg in [run_Lstar, run_Lsharp, run_KV]
]


@pytest.mark.parametrize("learning_alg,automaton_type,seed_val,num_states,input_size,output_size", TEST_CASES)
@pytest.mark.timeout(5)
def test_learning_algs_on_small_random_automata_exhaustive(learning_alg, automaton_type, seed_val, num_states,
                                                            input_size, output_size):
    random.seed(seed_val)

    model = generate_random_deterministic_automata(
        automaton_type,
        num_states=num_states,
        input_alphabet_size=input_size,
        output_alphabet_size=output_size,
    )

    sul = AutomatonSUL(model)
    input_alphabet = model.get_input_alphabet()

    eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=num_states * 10, walk_len=15)

    learned_model = learning_alg(input_alphabet, sul, eq_oracle, automaton_type=automaton_type, print_level=0)

    assert learned_model.is_minimal()
    assert bisimilar(model, learned_model)
