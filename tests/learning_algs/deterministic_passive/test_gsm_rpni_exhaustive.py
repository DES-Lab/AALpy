import pytest

from aalpy.learning_algs.deterministic_passive.GsmRPNI import GsmRPNI
from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import EDSM_score, SimpleScoreCalculation
from aalpy.utils import generate_random_deterministic_automata
from aalpy.utils.HelperFunctions import dfa_from_moore
from aalpy.utils.Sampling import get_complete_sample
from aalpy.utils.ModelChecking import bisimilar

pytestmark = pytest.mark.exhaustive

# Mirrors tests/learning_algs/resetless/test_hW_exhaustive.py: sweep many (states, inputs, outputs, seed)
# combinations for each deterministic automaton type and check that GSM-RPNI and EDSM (general passive
# state merging with the classic EDSM score), given a complete (state-cover x characterization-set)
# sample, both reconstruct a model bisimilar to the ground truth.
SEEDS = list(range(30))
MODEL_SIZES = [
    (2, 2, 3),
    (3, 3, 2),
    (4, 3, 2),
    (5, 3, 3),
    (6, 2, 3),
    (6, 3, 2),
    (10, 2, 3),
    (10, 2, 4),
    (20, 2, 3),
    (30, 2, 4),
    (30, 3, 2),
    (50, 3, 5),
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


def complete_data_set(automaton, automaton_type):
    """
    Builds a complete list of (input_sequence, output_label) pairs for GSM-RPNI, labeling every prefix
    of every sequence in the state-cover x characterization-set complete sample with its own final-step
    output. Labeling only the full sequences (and not each of their prefixes) would, for mealy machines,
    leave individual transitions along the way unobserved as a "final" label and thus undetermined.
    """
    automaton.compute_prefixes()

    data = []
    if automaton_type in ('dfa', 'moore'):
        data.append(((), automaton.initial_state.output))

    seen = set()
    for seq in get_complete_sample(automaton):
        if not seq:
            continue
        automaton.reset_to_initial()
        outputs = automaton.execute_sequence(automaton.initial_state, seq)
        for k in range(1, len(seq) + 1):
            prefix = tuple(seq[:k])
            if prefix in seen:
                continue
            seen.add(prefix)
            data.append((prefix, outputs[k - 1]))

    return data


def complete_mealy_io_traces(automaton):
    """
    Builds full input/output traces (as expected by run_GSM's 'io_traces' format for mealy behavior)
    covering the same state-cover x characterization-set sample as complete_data_set.
    """
    automaton.compute_prefixes()

    traces = []
    for seq in get_complete_sample(automaton):
        if not seq:
            continue
        automaton.reset_to_initial()
        outputs = automaton.execute_sequence(automaton.initial_state, seq)
        traces.append(list(zip(seq, outputs)))

    return traces


def run_edsm(automaton, automaton_type):
    """
    Runs general passive state merging with the classic EDSM score function on a complete sample of
    the given automaton, returning a model of the same automaton_type as the ground truth.
    """
    score_calc = SimpleScoreCalculation(score_function=EDSM_score())

    if automaton_type == 'mealy':
        traces = complete_mealy_io_traces(automaton)
        return run_GSM(traces, output_behavior='mealy', transition_behavior='deterministic',
                        score_calc=score_calc, data_format='io_traces')

    data = complete_data_set(automaton, automaton_type)
    learned_moore = run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
                             score_calc=score_calc, data_format='labeled_sequences')
    return dfa_from_moore(learned_moore) if automaton_type == 'dfa' else learned_moore


@pytest.mark.parametrize("automaton_type,seed_val,num_states,input_size,output_size", TEST_CASES)
@pytest.mark.timeout(30)
def test_gsm_rpni_seed_exhaustive(automaton_type, seed_val, num_states, input_size, output_size):
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

    data = complete_data_set(model, automaton_type)

    rpni_learned_model = GsmRPNI(data, automaton_type, print_info=False).run_rpni()

    assert rpni_learned_model is not None
    assert rpni_learned_model.is_minimal()
    assert bisimilar(model, rpni_learned_model)
    assert len(rpni_learned_model.states) == len(model.states)

    edsm_learned_model = run_edsm(model, automaton_type)

    assert edsm_learned_model.is_minimal()
    assert bisimilar(model, edsm_learned_model)
    assert len(edsm_learned_model.states) == len(model.states)
