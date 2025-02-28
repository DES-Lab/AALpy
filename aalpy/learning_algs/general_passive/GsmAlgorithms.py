from typing import Dict, Union

from aalpy import DeterministicAutomaton, Onfsm, NDMooreMachine
from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
from aalpy.learning_algs.general_passive.Instrumentation import ProgressReport
from aalpy.learning_algs.general_passive.GsmNode import GsmNode
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation, hoeffding_compatibility, \
    ScoreWithKTail
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

    print_level = ProgressReport(1) if print_info else None

    def EDSM_score(part: Dict[GsmNode, GsmNode]):
        nr_partitions = len(set(part.values()))
        nr_merged = len(part)
        return nr_merged - nr_partitions

    score = ScoreCalculation(score_function=EDSM_score)

    internal_automaton_type = 'moore' if automaton_type != 'mealy' else automaton_type

    learned_model = run_GSM(data, output_behavior=internal_automaton_type,
                            transition_behavior="deterministic",
                            score_calc=score, data_format='labeled_sequences', instrumentation=print_level)

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


def run_k_tails(data, automaton_type, k, input_completeness=None, print_info=True) -> Union[
    Onfsm, NDMooreMachine, None]:
    """
    Runs k-tails.

    Args:

        data: sequence of input-output traces

        automaton_type: either 'mealy' or 'moore'. Note that the data has to be prefix-closed, and the resulting model
                        could be non-deterministic.

        k: depth until which to check node compatibility

        input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be input incomplete,

        sink_state will lead all undefined inputs form some state to the sink state, whereas self_loop will simply create

        a self loop. In case of Mealy learning output of the added transition will be 'epsilon'.

        print_info: print learning progress and runtime information

    Returns:

        Model conforming to the data such that future compatibility is checked only until the depth of k.

    """
    assert automaton_type in {'mealy', 'moore'}
    assert input_completeness in {None, 'self_loop', 'sink_state'}

    print_level = ProgressReport(1) if print_info else None

    internal_automaton_type = 'moore' if automaton_type != 'mealy' else automaton_type

    score = ScoreWithKTail(ScoreCalculation(GsmNode.deterministic_compatible), k)

    learned_model = run_GSM(data, output_behavior=internal_automaton_type,
                            transition_behavior="nondeterministic",
                            score_calc=score, data_format='io_traces', instrumentation=print_level)

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


def run_Alergia_EDSM(data, automaton_type, eps=0.05, print_info=False):
    """
    Run IoAlergia with EDSM on provided data.

    Args:

        data: [[O,(I,O),(I,O)...], [O,(I,O), (I, O)_,...],..,] if learning MDPs,
        or [[I,O,I,O...], [I,O_,...],..,] if learning SMMs (I represents input, O output).
        Note that in whole data first symbol of each entry should be the same (Initial output of the MDP).

        eps: epsilon value if you are using default HoeffdingCompatibility.

        automaton_type: either 'mdp' if you wish to learn an MDP, or 'smm' if you want to learn stochastic Mealy machine

        print_info: default False

    Returns:

        A Mdp or SMM
    """

    assert automaton_type in {'mdp', 'smm'}

    print_level = ProgressReport(1) if print_info else None

    class IOAlergiaWithEDSM(ScoreCalculation):
        def __init__(self, epsilon):
            super().__init__()
            self.ioa_compatibility = hoeffding_compatibility(epsilon)
            self.evidence = 0

        def reset(self):
            self.evidence = 0

        def local_compatibility(self, a: GsmNode, b: GsmNode):
            self.evidence += 1
            return self.ioa_compatibility(a, b)

        def score_function(self, part: dict[GsmNode, GsmNode]):
            return self.evidence

    output_behaviour = 'moore' if automaton_type == 'mdp' else 'mealy'

    learned_model = run_GSM(data, output_behavior=output_behaviour, transition_behavior="stochastic",
                            score_calc=IOAlergiaWithEDSM(eps),
                            compatibility_on_pta=True, compatibility_on_futures=True,
                            instrumentation=print_level, data_format='io_traces')

    return learned_model
