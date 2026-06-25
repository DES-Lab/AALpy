import pytest

from aalpy import generate_random_deterministic_automata, bisimilar, run_hW, RandomhWOracle, RandomWphWOracle, \
    AutomatonSUL
from aalpy.SULs import MealySUL

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
]


@pytest.mark.parametrize("automaton_type,seed_val,num_states,input_size,output_size", TEST_CASES)
@pytest.mark.timeout(5)
def test_hw_seed(automaton_type, seed_val, num_states, input_size, output_size):
    from random import seed

    seed(seed_val)

    model = generate_random_deterministic_automata(
        automaton_type,
        num_states=num_states,
        input_alphabet_size=input_size,
        output_alphabet_size=output_size,
    )
    if not model.is_minimal():
        pytest.skip(f"seed {seed_val} does not produce a minimal model")
    if not model.is_strongly_connected():
        pytest.skip(
            f"seed {seed_val} does not produce a strongly connected model "
            f"for states={num_states}, inputs={input_size}, outputs={output_size}"
        )

    sul = MealySUL(model)
    input_alphabet = model.get_input_alphabet()


    learned_model = run_hW(input_alphabet,
                     sul,
                     RandomhWOracle(num_testing_steps=1000 * num_states, reset_testing_counter=True),
                     automaton_type=automaton_type,
                     query_for_initial_state=True)

    assert learned_model.is_minimal()
    assert bisimilar(model, learned_model)


STRATEGY_TEST_CASES = [
    pytest.param(
        oracle_name,
        seed_val,
        num_states,
        input_size,
        output_size,
        id=f"oracle={oracle_name}-states={num_states}-inputs={input_size}-outputs={output_size}-seed={seed_val}",
    )
    for num_states, input_size, output_size in [(3, 2, 2), (5, 3, 3), (10, 2, 3)]
    for seed_val in range(10)
    for oracle_name in ['random', 'wp']
]


@pytest.mark.parametrize("oracle_name,seed_val,num_states,input_size,output_size", STRATEGY_TEST_CASES)
@pytest.mark.timeout(5)
def test_hw_eq_oracle(oracle_name, seed_val, num_states, input_size, output_size):
    from random import seed

    seed(seed_val)

    model = generate_random_deterministic_automata(
        'mealy',
        num_states=num_states,
        input_alphabet_size=input_size,
        output_alphabet_size=output_size,
    )
    if not model.is_minimal() or not model.is_strongly_connected():
        pytest.skip(f"seed {seed_val} does not produce a minimal, strongly connected model")

    sul = AutomatonSUL(model)
    input_alphabet = model.get_input_alphabet()

    if oracle_name == 'wp':
        eq_oracle = RandomWphWOracle(random_walk_length=4 * num_states,
                                     num_test_origin_states=4 * num_states)
    else:
        eq_oracle = RandomhWOracle(num_testing_steps=1000 * num_states)

    learned_model = run_hW(input_alphabet,
                           sul,
                           eq_oracle,
                           automaton_type='mealy',
                           query_for_initial_state=True,
                           print_level=0)

    assert learned_model.is_minimal()
    assert bisimilar(model, learned_model)


def test_hw_uses_user_provided_h_and_w():
    from aalpy import bisimilar, generate_random_deterministic_automata, run_hW, AutomatonSUL
    from random import seed
    seed(1)

    model = generate_random_deterministic_automata(
        'mealy',
        num_states=100,
        input_alphabet_size=4,
        output_alphabet_size=4,
    )

    sul = AutomatonSUL(model)
    input_alphabet = model.get_input_alphabet()

    char_set = model.compute_characterization_set()

    assert model.is_minimal() and model.is_minimal()

    learned_model = run_hW(input_alphabet,
                           sul,
                           RandomhWOracle(num_testing_steps=2000),
                           automaton_type='mealy',
                           provided_characterization_set=char_set,
                           query_for_initial_state=True)

    assert learned_model.is_minimal()
    assert bisimilar(model, learned_model)

