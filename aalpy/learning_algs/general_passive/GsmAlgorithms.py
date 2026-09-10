# Convenience wrappers around run_GSM implementing well-known passive learning
# algorithms: EDSM, k-tails, and Alergia/IoAlergia (with EDSM-style scoring).
from collections import defaultdict
from functools import partial

from aalpy import DeterministicAutomaton, Onfsm, NDMooreMachine
from aalpy.base import Automaton
from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
from aalpy.learning_algs.general_passive.DataHandler import CountOnPTADataHandler
from aalpy.learning_algs.general_passive.Instrumentation import ProgressReport
from aalpy.learning_algs.general_passive.GsmNode import GsmNode, unknown_output
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import SimpleScoreCalculation, ScoreWithKTail, ScoreIOAlergiaWithEDSM
from aalpy.utils.HelperFunctions import dfa_from_moore, mc_format_to_mdp, mc_from_mdp


def run_EDSM(data: list, automaton_type: str, input_completeness: str | None = None,
             print_info: bool = True) -> DeterministicAutomaton | None:
    """
    Run Evidence Driven State Merging.

    :param list data: sequence of input sequences and corresponding label, e.g. [[(i1,i2,i3, ...), label], ...]
    :param str automaton_type: either 'dfa', 'mealy', 'moore'. Note that for 'mealy' machine learning, data has to be prefix-closed.
    :param str | None input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be input incomplete,
        sink_state will lead all undefined inputs form some state to the sink state, whereas self_loop will simply create
        a self loop. In case of Mealy learning output of the added transition will be 'epsilon'.
    :param bool print_info: print learning progress and runtime information
    :return DeterministicAutomaton | None: Model conforming to the data, or None if data is non-deterministic.
    """
    assert automaton_type in {'dfa', 'mealy', 'moore'}
    assert input_completeness in {None, 'self_loop', 'sink_state'}

    print_level = ProgressReport(1) if print_info else None

    def EDSM_score(part: dict[GsmNode, GsmNode]) -> int:
        reverse_partition = defaultdict(list)
        for original_node, resulting_node in part.items():
            reverse_partition[resulting_node].append(original_node)
        evidence = 0
        for node, contributing_nodes in reverse_partition.items():
            if node.get_prefix_output() is unknown_output:
                continue  # No evidence whatsoever
            evidence -= 1  # subtract self-comparison
            for contributing_node in contributing_nodes:
                if contributing_node.get_prefix_output() is not unknown_output:
                    evidence += 1
        return evidence

    score = SimpleScoreCalculation(local_compatibility=GsmNode.deterministic_compatible, score_function=EDSM_score)

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


def run_k_tails(data: list, automaton_type: str, k: int, input_completeness: str | None = None,
                print_info: bool = True) -> Onfsm | NDMooreMachine | None:
    """
    Runs k-tails.

    :param list data: sequence of input-output traces
    :param str automaton_type: either 'mealy' or 'moore'. Note that the data has to be prefix-closed, and the resulting model
        could be non-deterministic.
    :param int k: depth until which to check node compatibility
    :param str | None input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be input incomplete,
        sink_state will lead all undefined inputs form some state to the sink state, whereas self_loop will simply create
        a self loop. In case of Mealy learning output of the added transition will be 'epsilon'.
    :param bool print_info: print learning progress and runtime information
    :return Onfsm | NDMooreMachine | None: Model conforming to the data such that future compatibility is checked only
        until the depth of k.
    """
    assert automaton_type in {'mealy', 'moore'}
    assert input_completeness in {None, 'self_loop', 'sink_state'}

    print_level = ProgressReport(1) if print_info else None

    internal_automaton_type = 'moore' if automaton_type != 'mealy' else automaton_type

    score = ScoreWithKTail(SimpleScoreCalculation(GsmNode.deterministic_compatible), k)

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

def run_Alergia_GSM(data: list, automaton_type: str, eps: float = 0.05, compat_on_pta_trans: bool = True, compat_on_pta_count: bool = True, edsm: bool = False, print_info: bool = False) -> Automaton:
    """
    Run IOAlergia on provided data. Also supports variants that
    - use data more extensively than the original
    - use EDSM based scoring

    :param list data: [[O,(I,O),(I,O)...], [O,(I,O), (I, O)_,...],...,] if learning MDPs,
        or [[I,O,I,O...], [I,O_,...],...,] if learning SMMs (I represent input, O output), or [[O, O, O], ...] if
        learning Markov chains.
        Note that when learning MDPs and MCs the first symbol of each entry should be the same (Initial output).
    :param str automaton_type: either 'mdp' if you wish to learn an MDP, or 'smm' if you want to learn stochastic Mealy machine
    :param float eps: epsilon value if you are using default HoeffdingCompatibility.
    :param compat_on_pta_trans: evaluate compatibility criterion only on transitions present in the respective PTA nodes
    :param compat_on_pta_count: evaluate compatibility criterion using counts from the respective PTA nodes
    :param bool edsm: enable EDSM based scoring
    :param bool print_info: default False
    :return Automaton: A Mc, Mdp or SMM
    """

    at_types = ['mc', 'mdp', 'smm']
    if automaton_type not in at_types:
        raise ValueError(f"automaton_type {automaton_type} not in {at_types}")

    if not compat_on_pta_trans and compat_on_pta_count:
        raise ValueError("compat_on_pta must be set if compat_on_pta_data is")

    instrumentation = ProgressReport(1) if print_info else None

    output_behaviour = 'moore' if automaton_type != 'smm' else 'mealy'

    learning_data = data if automaton_type != 'mc' else mc_format_to_mdp(data)

    learned_model = run_GSM(
        learning_data,
        output_behavior=output_behaviour,
        transition_behavior="stochastic",
        data_handler=CountOnPTADataHandler(),
        score_calc=ScoreIOAlergiaWithEDSM(eps, compat_on_pta_trans, compat_on_pta_count, edsm),
        instrumentation=instrumentation,
        data_format='io_traces',
    )

    if automaton_type == 'mc':
        learned_model = mc_from_mdp(learned_model)

    return learned_model

run_Alergia_EDSM = partial(run_Alergia_GSM, edsm=True)