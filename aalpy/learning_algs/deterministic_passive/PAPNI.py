from aalpy.utils import is_balanced
from aalpy.automata.Vpa import vpa_from_dfa_representation

def run_PAPNI(data, vpa_alphabet, algorithm='edsm', print_info=True):
    """
    Run PAPNI, a deterministic passive model learning algorithm of deterministic pushdown automata.
    Resulting model conforms to the provided data.

    Args:

        data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
        vpa_alphabet:  grouping of alphabet elements to call symbols, return symbols, and internal symbols. Call symbols
        push to stack, return symbols pop from stack, and internal symbols do not affect the stack.
        algorithm: either 'gsm' for classic RPNI or 'edsm' for evidence driven state merging variant of RPNI
        print_info: print learning progress and runtime information

    Returns:

        VPA conforming to the data, or None if data is non-deterministic.
    """
    from aalpy.learning_algs import run_EDSM, run_RPNI
    assert algorithm in {'gsm', 'classic', 'edsm'}

    # preprocess input sequences to keep track of stack
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
