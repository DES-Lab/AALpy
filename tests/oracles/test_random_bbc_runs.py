import random

import pytest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_KV, run_Lsharp, run_Lstar
from aalpy.model_checking_oracles import IUOBugDfaModelCheckingOracle
from aalpy.oracles import BBCEqOracle, WMethodEqOracle
from aalpy.utils.AutomatonGenerators import generate_random_dfa, generate_random_mealy_machine
from aalpy.utils.ModelChecking import bisimilar


# Less exhaustive version of the randomly-generated black-box checking tests from
# tests/oracles/test_random_bbc_runs_exhaustive.py. Sweeps many combinations of
# (states, inputs, outputs, properties, seed). Restricted to Mealy machine SULs because AALpy
# currently only has concrete model checking oracle instances (found in aalpy.model_checking_oracles))
# that work for Mealy machines.
# Checks that performing learning with several learning algorithms and the BBCEqOracle equivalence
# oracle finds counterexamples for precisely the properties that are violated by the SUL.


SEEDS = list(range(6))
MODEL_SIZES_AND_NUM_PROPS = [
    (2, 2, 2, 4),
    (3, 2, 3, 6),
    (4, 3, 2, 8),
    (6, 2, 3, 10)
]

TEST_CASES = [
    pytest.param(
        learning_alg,
        seed_val,
        num_states,
        input_size,
        output_size,
        num_props,
        id=f"{learning_alg.__name__}-states={num_states}-inputs={input_size}-outputs={output_size}-props={num_props}-seed={seed_val}",
    )
    for num_states, input_size, output_size, num_props in MODEL_SIZES_AND_NUM_PROPS
    for seed_val in SEEDS
    for learning_alg in [run_Lstar, run_Lsharp, run_KV]
]


def dfa_input_from_mealy_input(input: Any) -> Tuple[str,Any]:
    return ("I", input)

def dfa_output_from_mealy_output(output: Any) -> Tuple[str,Any]:
    return ("O", output)

def is_dfa_input(letter: Tuple[str,Any]) -> bool:
    return letter[0] == "I"

def mealy_letter_from_dfa_letter(letter: Tuple[str,Any]) -> Any:
    return letter[1]

def generate_random_property_oracle(num_states: int, alphabet: list, num_accepting_states: int):
    bug_dfa = generate_random_dfa(
        num_states=num_states,
        alphabet=alphabet,
        num_accepting_states=num_accepting_states,
        compute_prefixes=False,
        ensure_minimality=False)
    bug_dfa.make_input_complete()

    return IUOBugDfaModelCheckingOracle(
        bug_dfa=bug_dfa,
        mealy_input_to_dfa_input=dfa_input_from_mealy_input,
        mealy_output_to_dfa_output=dfa_output_from_mealy_output,
        is_dfa_input=is_dfa_input,
        dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
    )


@pytest.mark.parametrize("learning_alg,seed_val,num_states,input_size,output_size,num_props", TEST_CASES)
@pytest.mark.timeout(5)
def test_bbc_on_small_random_automata_with_few_properties_exhaustive(learning_alg, seed_val, num_states,
                                                                     input_size, output_size, num_props):
    random.seed(seed_val)

    input_alphabet = [f"I{i}" for i in range(input_size)]
    output_alphabet = [f"O{i}" for i in range(output_size)]

    mealy = generate_random_mealy_machine(
        num_states=num_states,
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet)

    sul = AutomatonSUL(mealy)

    eq_oracle = WMethodEqOracle(input_alphabet, sul, len(mealy.states) + 1)

    prop_labels_to_counterexample_map = dict()
    def violation_callback(label: str, cex: tuple) -> None:
        nonlocal prop_labels_to_counterexample_map

        assert label not in prop_labels_to_counterexample_map
        prop_labels_to_counterexample_map[label] = cex

    properties = {
        f"prop_{i}": generate_random_property_oracle(
            num_states=num_states,
            alphabet=[dfa_input_from_mealy_input(input) for input in input_alphabet] + \
                     [dfa_output_from_mealy_output(output) for output in output_alphabet],
            num_accepting_states=num_states // 2)
        for i in range(num_props)
    }

    bbc_oracle = BBCEqOracle(
        eq_oracle=eq_oracle,
        property_oracles=properties,
        property_violation_callback=violation_callback
    )

    learned_mealy = learning_alg(input_alphabet, sul, bbc_oracle, automaton_type="mealy", print_level=0)

    assert learned_mealy.is_minimal()
    assert bisimilar(mealy, learned_mealy)

    num_rightfully_violated_properties = 0
    for label, prop in properties.items():
        cex = prop.find_cex(mealy)

        if cex is None:
            assert label not in prop_labels_to_counterexample_map
        else:
            assert label in prop_labels_to_counterexample_map
            stored_cex = prop_labels_to_counterexample_map[label]

            num_rightfully_violated_properties += 1
    assert num_rightfully_violated_properties == len(prop_labels_to_counterexample_map)