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
    (5, 5, 5),
    (6, 2, 3),
    (6, 4, 2),
    (10, 2, 3),
    (10, 2, 4),
    (10, 2, 2),
    (10, 4, 3),
    (12, 2, 3),
    (12, 2, 4),
    (12, 2, 3),
    (15, 2, 4),
    (15, 4, 2),
    (20, 4, 4),
]

TEST_CASES = [
    pytest.param(
        seed_val,
        num_states,
        input_size,
        output_size,
        id=f"states={num_states}-inputs={input_size}-outputs={output_size}-seed={seed_val}",
    )
    for num_states, input_size, output_size in MODEL_SIZES
    for seed_val in SEEDS
]


@pytest.mark.parametrize("seed_val,num_states,input_size,output_size", TEST_CASES)
@pytest.mark.timeout(10)
def test_hw_seed(seed_val, num_states, input_size, output_size):
    from random import seed

    seed(seed_val)

    model = generate_random_deterministic_automata(
        'mealy',
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
                     reset_testing_counter=True,
                     query_for_initial_state=True)

    assert learned_model.is_minimal()

    print(learned_model)
    print(model)
    assert bisimilar(model, learned_model)
