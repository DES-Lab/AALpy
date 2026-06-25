import pytest

from aalpy import generate_random_deterministic_automata, bisimilar, AutomatonSUL, run_Lstar, run_KV, \
    RandomWMethodEqOracle
from aalpy.learning_algs import run_Lsharp

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
        id=f"states={num_states}-inputs={input_size}-outputs={output_size}-seed={seed_val}-automaton_type={automaton_type}",
    )
    for num_states, input_size, output_size in MODEL_SIZES
    for seed_val in SEEDS
    for automaton_type in ['dfa', 'moore', 'mealy']
    for learning_alg in [run_Lstar, run_Lsharp, run_KV]
]


@pytest.mark.parametrize("learning_alg,automaton_type,seed_val,num_states,input_size,output_size", TEST_CASES)
@pytest.mark.timeout(5)
def test_learning_algs(learning_alg: callable, automaton_type, seed_val, num_states, input_size, output_size):
    from random import seed

    seed(seed_val)

    model = generate_random_deterministic_automata(
        automaton_type,
        num_states=num_states,
        input_alphabet_size=input_size,
        output_alphabet_size=output_size,
    )

    sul = AutomatonSUL(model)
    input_alphabet = model.get_input_alphabet()

    eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=num_states * 10, walk_len=15)

    learned_model = learning_alg(input_alphabet, sul, eq_oracle, automaton_type=automaton_type, print_level = 0)

    assert learned_model.is_minimal()

    print(learned_model)
    print(model)
    assert bisimilar(model, learned_model)

