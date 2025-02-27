from typing import Union

from aalpy.base import DeterministicAutomaton
from aalpy.learning_algs.deterministic_passive.ClassicRPNI import ClassicRPNI
from aalpy.learning_algs.deterministic_passive.GsmRPNI import GsmRPNI


def run_RPNI(data, automaton_type, algorithm='gsm',
             input_completeness=None, print_info=True) -> Union[DeterministicAutomaton, None]:
    """
    Run RPNI, a deterministic passive model learning algorithm.
    Resulting model conforms to the provided data.
    For more information on RPNI, check out AALpy' Wiki:
    https://github.com/DES-Lab/AALpy/wiki/RPNI---Passive-Deterministic-Automata-Learning

    Args:

        data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
        automaton_type: either 'dfa', 'mealy', 'moore'. Note that for 'mealy' machine learning, data has to be prefix-closed.
        algorithm: either 'gsm' (generalized state merging) or 'classic' for base RPNI implementation. GSM is much faster and less resource intensive.
        input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be input incomplete,
        sink_state will lead all undefined inputs form some state to the sink state, whereas self_loop will simply create
        a self loop. In case of Mealy learning output of the added transition will be 'epsilon'.
        print_info: print learning progress and runtime information

    Returns:

        Model conforming to the data, or None if data is non-deterministic.
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


def run_PAPNI(data, vpa_alphabet, algorithm='edsm', print_info=True):
    """
    Run PAPNI, a deterministic passive model learning algorithm of deterministic pushdown automata.
    Resulting model conforms to the provided data.

    Args:

        data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
        vpa_alphabet:  grouping of alphabet elements to call symbols, return symbols, and internal symbols. Call symbols
        push to stack, return symbols pop from stack, and internal symbols do not affect the stack.
        algorithm: either 'gsm' for classic RPNI or 'edsm' for evidence driven state merging variant of RPNI
        GSM is much faster and less resource intensive.
        print_info: print learning progress and runtime information

    Returns:

        VPA conforming to the data, or None if data is non-deterministic.
    """
    from aalpy.utils import is_balanced
    from aalpy.automata.Vpa import vpa_from_dfa_representation
    from aalpy.learning_algs import run_EDSM

    assert algorithm in {'gsm', 'classic', 'edsm'}

    # preprocess input sequances to keep track of stack
    papni_data = []
    for input_seq, label in data:
        # if input sequance is not balanced we do not consider it (it would lead to error state anyway)
        if not is_balanced(input_seq, vpa_alphabet):
            continue

        # for each sequance keep track of the stack, and when pop/return element is observed encode it along with the
        # current top of stack. This keeps track of stack during execution
        processed_sequance = []
        stack = []

        for input_symbol in input_seq:
            input_element = input_symbol
            # if call/push symbol push to stack
            if input_symbol in vpa_alphabet.call_alphabet:
                stack.append(input_symbol)
            # if return/pop symbol pop from stack and add it to the input data
            if input_symbol in vpa_alphabet.return_alphabet:
                top_of_stack = stack.pop()
                input_element = (input_symbol, top_of_stack)
            processed_sequance.append(input_element)

        papni_data.append((processed_sequance, label))

    # instantiate and run PAPNI as base RPNI with stack-aware data
    if algorithm != 'edsm':
        learned_model = run_RPNI(papni_data, automaton_type='dfa', algorithm=algorithm, print_info=print_info)
    else:
        learned_model = run_EDSM(papni_data, automaton_type='dfa', print_info=print_info)

    # convert intermediate DFA representation to VPA
    learned_model = vpa_from_dfa_representation(learned_model, vpa_alphabet)

    return learned_model
