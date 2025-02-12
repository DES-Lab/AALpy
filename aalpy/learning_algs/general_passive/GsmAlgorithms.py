from typing import Dict, Union

from aalpy import DeterministicAutomaton
from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
from aalpy.learning_algs.general_passive.Instrumentation import ProgressReport
from aalpy.learning_algs.general_passive.Node import Node
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation
from aalpy.utils.HelperFunctions import dfa_from_moore


def run_EDSM(data, automaton_type, input_completeness=None, print_info=True) -> Union[DeterministicAutomaton, None]:
    """
    Run Evidence Driven State Merging.

    Args:
        data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
        automaton_type: either 'dfa', 'mealy', 'moore'. Note that for 'mealy' machine learning, data has to be prefix-closed.
        input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be input incomplete,
        sink_state will lead all undefined inputs form some state to the sink state, whereas self_loop will simply create
        a self loop. In case of Mealy learning output of the added transition will be 'epsilon'.
        print_info: print learning progress and runtime information

    Returns:

        Model conforming to the data, or None if data is non-deterministic.

    """
    assert automaton_type in {'dfa', 'mealy', 'moore'}
    assert input_completeness in {None, 'self_loop', 'sink_state'}

    def EDSM_score(part: Dict[Node, Node]):
        nr_partitions = len(set(part.values()))
        nr_merged = len(part)
        return nr_merged - nr_partitions

    score = ScoreCalculation(score_function=EDSM_score)

    internal_automaton_type = 'moore' if automaton_type != 'mealy' else automaton_type

    learned_model = run_GSM(data, output_behavior=internal_automaton_type,
                            transition_behavior="deterministic",
                            score_calc=score, instrumentation=ProgressReport(2))

    if automaton_type == 'dfa':
        learned_model = dfa_from_moore(learned_model)

    if not learned_model.is_input_complete():
        if not input_completeness:
            if print_info:
                print('Warning: Learned Model is not input complete (inputs not defined for all states). '
                      'Consider calling .make_input_complete()')
        else:
            if print_info:
                print(f'Learned model was not input complete. Adapting it with {input_completeness} transitions.')
            learned_model.make_input_complete(input_completeness)

    return learned_model
