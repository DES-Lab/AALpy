# Convenience wrappers around run_GSM implementing well-known passive learning
# algorithms: EDSM, k-tails, and Alergia/IoAlergia (with EDSM-style scoring).
from collections import defaultdict

from aalpy import DeterministicAutomaton, Onfsm, NDMooreMachine
from aalpy.base import Automaton
from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
from aalpy.learning_algs.general_passive.Instrumentation import ProgressReport
from aalpy.learning_algs.general_passive.GsmNode import GsmNode
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation, hoeffding_compatibility, \
    ScoreWithKTail
from aalpy.utils.HelperFunctions import dfa_from_moore


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
            if node.get_prefix_output() is None:
                continue  # No evidence whatsoever
            evidence -= 1  # subtract self-comparison
            for contributing_node in contributing_nodes:
                if contributing_node.get_prefix_output() is not None:
                    evidence += 1
        return evidence

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


def run_Alergia_EDSM(data: list, automaton_type: str, eps: float = 0.05, print_info: bool = False) -> Automaton:
    """
    Run IOAlergia with EDSM on provided data.

    :param list data: [[O,(I,O),(I,O)...], [O,(I,O), (I, O)_,...],...,] if learning MDPs,
        or [[I,O,I,O...], [I,O_,...],...,] if learning SMMs (I represent input, O output), or [[O, O, O], ...] if
        learning Markov chains.
        Note that when learning MDPs and MCs the first symbol of each entry should be the same (Initial output).
    :param float eps: epsilon value if you are using default HoeffdingCompatibility.
    :param str automaton_type: either 'mdp' if you wish to learn an MDP, or 'smm' if you want to learn stochastic Mealy machine
    :param bool print_info: default False
    :return Automaton: A Mc, Mdp or SMM
    """
    from aalpy.utils.HelperFunctions import mc_format_to_mdp, mc_from_mdp

    # TODO: This needs to be reworked...
    raise NotImplementedError

    assert automaton_type in {'mc', 'mdp', 'smm',}

    print_level = ProgressReport(1) if print_info else None

    class IOAlergiaWithEDSM(ScoreCalculation):
        """ScoreCalculation combining IoAlergia's Hoeffding compatibility with an EDSM-style evidence score."""

        def __init__(self, epsilon: float) -> None:
            """
            Create an IoAlergia+EDSM score calculation.

            :param float epsilon: Confidence parameter for the Hoeffding compatibility check.
            """
            super().__init__()
            self.ioa_compatibility = hoeffding_compatibility(epsilon)
            self.evidence = 0

        def initialize_merge(self, red: GsmNode, blue: GsmNode) -> bool | None:
            """
            Reset the accumulated evidence counter.
            """
            self.evidence = 0

        def local_compatibility(self, a: GsmNode, b: GsmNode) -> bool:
            """
            Check local compatibility of two nodes, accumulating evidence for the score function.

            :param GsmNode a: First node.
            :param GsmNode b: Second node.
            :return bool: True if the nodes are compatible according to the Hoeffding bound.
            """
            self.evidence += 1
            return self.ioa_compatibility(a, b)

        def score_function(self, part: dict[GsmNode, GsmNode]) -> int:
            """
            Compute the score of a merge partition as the accumulated evidence.

            :param dict[GsmNode, GsmNode] part: Mapping of original nodes to their merged partition representative.
            :return int: The accumulated evidence count.
            """
            return self.evidence

    output_behaviour = 'moore' if automaton_type != 'smm' else 'mealy'

    learning_data = data if automaton_type != 'mc' else mc_format_to_mdp(data)

    learned_model = run_GSM(learning_data, output_behavior=output_behaviour, transition_behavior="stochastic",
                            score_calc=IOAlergiaWithEDSM(eps),
                            compatibility_on_pta=True, compatibility_on_futures=True,
                            instrumentation=print_level, data_format='io_traces')

    if automaton_type == 'mc':
        learned_model = mc_from_mdp(learned_model)

    return learned_model
