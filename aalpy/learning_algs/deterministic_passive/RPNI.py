# Entry point for running RPNI, dispatching to either the classic or GSM implementation.
from aalpy.base import DeterministicAutomaton
from aalpy.learning_algs.deterministic_passive.ClassicRPNI import ClassicRPNI
from aalpy.learning_algs.deterministic_passive.GsmRPNI import GsmRPNI


def run_RPNI(data: list, automaton_type: str, algorithm: str = 'gsm',
             input_completeness: str | None = None, print_info: bool = True) -> DeterministicAutomaton | None:
    """
    Run RPNI, a deterministic passive model learning algorithm.
    Resulting model conforms to the provided data.
    For more information on RPNI, check out AALpy' Wiki:
    https://github.com/DES-Lab/AALpy/wiki/RPNI---Passive-Deterministic-Automata-Learning

    :param list data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
    :param str automaton_type: either 'dfa', 'mealy', 'moore'. Note that for 'mealy' machine learning, data has to
        be prefix-closed.
    :param str algorithm: either 'gsm' (generalized state merging) or 'classic' for base RPNI implementation. GSM is
        much faster and less resource intensive.
    :param str | None input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be
        input incomplete, sink_state will lead all undefined inputs form some state to the sink state, whereas
        self_loop will simply create a self loop. In case of Mealy learning output of the added transition will be
        'epsilon'.
    :param bool print_info: print learning progress and runtime information
    :return DeterministicAutomaton | None: Model conforming to the data, or None if data is non-deterministic.
    """
    assert algorithm in {'gsm', 'classic'}
    assert automaton_type in {'dfa', 'mealy', 'moore'}
    assert input_completeness in {None, 'self_loop', 'sink_state'}

    if algorithm == 'classic':
        rpni = ClassicRPNI(data, automaton_type, print_info)
    else:
        rpni = GsmRPNI(data, automaton_type, print_info)

    if rpni.root_node is None:
        print('Data provided to RPNI is not deterministic. Ensure that the data is deterministic, '
              'or consider using Alergia.')
        return None

    learned_model = rpni.run_rpni()

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
