import pytest

from aalpy import generate_random_deterministic_automata, bisimilar, run_hW
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
                     num_testing_steps=1000 * num_states,
                     automaton_type=automaton_type,
                     reset_testing_counter=True,
                     query_for_initial_state=True)

    assert learned_model.is_minimal()

    print(learned_model)
    print(model)
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
                           automaton_type='mealy',
                           provided_characterization_set=char_set,
                           num_testing_steps=2000,
                           reset_testing_counter=True,
                           query_for_initial_state=True)

    assert learned_model.is_minimal()
    assert bisimilar(model, learned_model)

