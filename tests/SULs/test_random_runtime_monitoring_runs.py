from itertools import chain
import random
from typing import Any, Tuple

import pytest

from aalpy.SULs import AutomatonSUL, MonitoringSUL
from aalpy.learning_algs import run_KV, run_Lsharp, run_Lstar
from aalpy.model_checking_oracles import IUOBugDfaModelCheckingOracle
from aalpy.oracles import WMethodEqOracle
from aalpy.property_monitors import IUOBugDfaMonitor
from aalpy.utils.AutomatonGenerators import generate_random_dfa, generate_random_mealy_machine
from aalpy.utils.ModelChecking import bisimilar


# Less exhaustive version of the randomly-generated runtime monitoring tests from
# tests/SULs/test_random_runtime_monitoring_runs_exhaustive.py. Sweeps many combinations of
# (states, inputs, outputs, properties, seed). Restricted to Mealy machine SULs because AALpy
# currently only has concrete property monitor instances (found in aalpy.property_monitors)) that
# work for Mealy machines.
# Checks that all counterexamples found by performing learning with several learning algorithms
# and the MonitoringSUL SUL only finds counterexamples for properties that are violated by the SUL.


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
        id=f'{learning_alg.__name__}-states={num_states}-inputs={input_size}-outputs={output_size}-props={num_props}-seed={seed_val}',
    )
    for num_states, input_size, output_size, num_props in MODEL_SIZES_AND_NUM_PROPS
    for seed_val in SEEDS
    for learning_alg in [run_Lstar, run_Lsharp, run_KV]
]


def dfa_input_from_mealy_input(input: Any) -> Tuple[str,Any]:
    return ('I', input)

def dfa_output_from_mealy_output(output: Any) -> Tuple[str,Any]:
    return ('O', output)

def is_dfa_input(letter: Tuple[str,Any]) -> bool:
    return letter[0] == 'I'

def mealy_letter_from_dfa_letter(letter: Tuple[str,Any]) -> Any:
    return letter[1]

def generate_random_IUO_bug_dfa_property_monitor(num_states: int,
                                                 alphabet: list,
                                                 num_accepting_states: int) -> IUOBugDfaMonitor:
    bug_dfa = generate_random_dfa(
        num_states=num_states,
        alphabet=alphabet,
        num_accepting_states=num_accepting_states,
        compute_prefixes=False,
        ensure_minimality=False)
    bug_dfa.make_input_complete()

    return IUOBugDfaMonitor(
        bug_dfa=bug_dfa,
        to_dfa_input=dfa_input_from_mealy_input,
        to_dfa_output=dfa_output_from_mealy_output
    )


@pytest.mark.parametrize("learning_alg,seed_val,num_states,input_size,output_size,num_props", TEST_CASES)
@pytest.mark.timeout(5)
def test_runtime_monitoring_on_small_random_automata_with_few_properties_exhaustive(learning_alg, seed_val, num_states,
                                                                                    input_size, output_size, num_props):
    random.seed(seed_val)

    input_alphabet = [f'I{i}' for i in range(input_size)]
    output_alphabet = [f'O{i}' for i in range(output_size)]

    mealy = generate_random_mealy_machine(
        num_states=num_states,
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet)

    sul = AutomatonSUL(mealy)

    eq_oracle = WMethodEqOracle(input_alphabet, sul, len(mealy.states) + 1)

    prop_labels_to_counterexample_map = dict()
    def violation_callback(label: str, cex: tuple) -> None:

        assert label not in prop_labels_to_counterexample_map
        prop_labels_to_counterexample_map[label] = cex

    properties = {
        f'prop_{i}': generate_random_IUO_bug_dfa_property_monitor(
            num_states=num_states,
            alphabet=[dfa_input_from_mealy_input(input) for input in input_alphabet] + \
                     [dfa_output_from_mealy_output(output) for output in output_alphabet],
            num_accepting_states=num_states // 2)
        for i in range(num_props)
    }

    mon_sul = MonitoringSUL(
        sul=sul,
        property_monitors=properties,
        property_violation_callback=violation_callback
    )

    learned_mealy = learning_alg(input_alphabet, mon_sul, eq_oracle, automaton_type='mealy', print_level=0)

    assert learned_mealy.is_minimal()
    assert bisimilar(mealy, learned_mealy)

    for label, prop in properties.items():
        bug_dfa = prop.bug_dfa

        if label in prop_labels_to_counterexample_map:
            cex = prop_labels_to_counterexample_map[label]

            io_pairs_trace = sul.io_query(cex)
            dfa_io_pairs_trace = [(dfa_input_from_mealy_input(i), dfa_output_from_mealy_output(o)) for (i, o) in io_pairs_trace]
            dfa_io_trace = tuple(chain(*[[i, o] for (i, o) in dfa_io_pairs_trace]))

            bug_dfa.reset_to_initial()
            encountered_accepting_state = False
            for dfa_a in dfa_io_trace:
                if bug_dfa.step(dfa_a):
                    encountered_accepting_state = True
                    break
            assert encountered_accepting_state
        else:
            bug_dfa_oracle = IUOBugDfaModelCheckingOracle(
                bug_dfa=bug_dfa,
                mealy_input_to_dfa_input=dfa_input_from_mealy_input,
                mealy_output_to_dfa_output=dfa_output_from_mealy_output,
                is_dfa_input=is_dfa_input,
                dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
            )
            assert bug_dfa_oracle.find_cex(mealy) is None