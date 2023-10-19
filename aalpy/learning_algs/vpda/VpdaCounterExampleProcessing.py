import re

from aalpy.SULs.AutomataSUL import SevpaSUL
from aalpy.base import SUL
from aalpy.utils.HelperFunctions import all_suffixes, all_prefixes
from aalpy.automata import Sevpa, SevpaState


def counterexample_successfully_processed(sul, cex, hypothesis):
    cex_outputs = sul.query(cex)
    hyp_outputs = hypothesis.execute_sequence(hypothesis.initial_state, cex)
    return cex_outputs[-1] == hyp_outputs[-1]


def longest_prefix_cex_processing(s_union_s_dot_a: list, cex: tuple, closedness='suffix'):
    """
    Suffix processing strategy found in Shahbaz-Groz paper 'Inferring Mealy Machines'.
    It splits the counterexample into prefix and suffix. The prefix is the longest element of the S union S.A that
    matches the beginning of the counterexample. By removing such prefixes from counterexample, no consistency check
    is needed.

    Args:

        s_union_s_dot_a: list of all prefixes found in observation table sorted from shortest to longest
        cex: counterexample
        closedness: either 'suffix' or 'prefix'. (Default value = 'suffix')
        s_union_s_dot_a: list:
        cex: tuple: counterexample

    Returns:

        suffixes to add to the E set

    """
    prefixes = s_union_s_dot_a
    prefixes.reverse()
    trimmed_suffix = None

    for p in prefixes:
        if p == cex[:len(p)]:
            trimmed_suffix = cex[len(p):]
            break

    trimmed_suffix = trimmed_suffix if trimmed_suffix else cex
    suffixes = all_suffixes(trimmed_suffix) if closedness == 'suffix' else all_prefixes(trimmed_suffix)
    suffixes.reverse()
    return suffixes


def rs_cex_processing(sul: SUL, cex: tuple, hypothesis: Sevpa, suffix_closedness=True, closedness='suffix'):
    """Riverst-Schapire counter example processing.

    Args:

        sul: system under learning
        cex: found counterexample
        hypothesis: hypothesis on which counterexample was found
        suffix_closedness: If true all suffixes will be added, else just one (Default value = True)
        closedness: either 'suffix' or 'prefix'. (Default value = 'suffix')
        sul: SUL: system under learning
        cex: tuple: counterexample

    Returns:

        suffixes to be added to the E set

    """
    cex_out = sul.query(cex)
    cex_input = list(cex)

    lower = 1
    upper = len(cex_input) - 2

    while True:
        hypothesis.reset_to_initial()
        mid = (lower + upper) // 2

        # arr[:n] -> first n values
        # arr[n:] -> last n values

        for s_p in cex_input[:mid]:
            hypothesis.step(s_p)
        s_bracket = hypothesis.current_state.prefix

        d = tuple(cex_input[mid:])
        mq = sul.query(s_bracket + d)

        if mq[-1] == cex_out[-1]:  # only check if the last element is the same as the cex
            lower = mid + 1
            if upper < lower:
                suffix = d[1:]
                break
        else:
            upper = mid - 1
            if upper < lower:
                suffix = d
                break

    hyp_sul = SevpaSUL(hypothesis)
    hyp_sul.query(('(',))
    word = transform_access_seq(hypothesis, hyp_sul.sevpa.stack)
    print(word)

    if suffix_closedness:
        suffixes = all_suffixes(suffix) if closedness == 'suffix' else all_prefixes(suffix)
        suffixes.reverse()
        suffix_to_query = suffixes
    else:
        suffix_to_query = [suffix]
    return suffix_to_query


def transform_access_seq(hypothesis: Sevpa, stack: []) -> list[str]:
    word = []
    pattern = r"(q\d+)(.*)"

    for i in range(1, len(stack)):  # skip the first element because it's the start of the stack '_
        stack_elem = stack[i]
        match = re.search(pattern, stack_elem)
        if match:
            from_state_id = match.group(1)      # the corresponding state where the stack element got pushed from
            call_letter = match.group(2)        # the call letter that was pushed on the stack
            print("From state:", from_state_id)
            print("Call letter:", call_letter)
            from_state = hypothesis.get_state_by_id(from_state_id)
            word.append(from_state.prefix)      # .prefix is the access sequence of the node in the classificationTree
            word.append(call_letter)
        else:
            assert False and print("Stack content does not follow convention")

        word.append(hypothesis.initial_state.prefix)
    return word
