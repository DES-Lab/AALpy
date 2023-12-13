from aalpy.base import SUL
from aalpy.utils.HelperFunctions import all_suffixes, all_prefixes


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


def rs_cex_processing(sul: SUL, cex: tuple, hypothesis, suffix_closedness=True, closedness='suffix',
                      is_vpa=False, lower=None, upper=None):
    """
    Riverst-Schapire counter example processing.

    Args:

        sul: system under learning
        cex: found counterexample
        hypothesis: hypothesis on which counterexample was found
        suffix_closedness: If true all suffixes will be added, else just one (Default value = True)
        closedness: either 'suffix' or 'prefix'. (Default value = 'suffix')
        sul: SUL: system under learning
        cex: tuple: counterexample
        is_vpa: system under learning behaves as a context free language
        upper: upper boarder for cex (from preprocessing), None will set it to 1
        lower: lower boarder for cex (from preprocessing), None will set it to  len(cex_input) - 2

    Returns:

        suffixes to be added to the E set

    """
    cex_out = sul.query(cex)
    cex_input = list(cex)

    lower = 1 if lower is None else lower
    upper = len(cex_input) - 2 if upper is None else upper

    while True:
        hypothesis.reset_to_initial()
        mid = (lower + upper) // 2

        # arr[:n] -> first n values
        # arr[n:] -> last n values

        for s_p in cex_input[:mid]:
            hypothesis.step(s_p)

        if not is_vpa:
            s_bracket = hypothesis.current_state.prefix
        else:
            s_bracket = tuple(hypothesis.transform_access_string(hypothesis.current_state))

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

    if suffix_closedness:
        suffixes = all_suffixes(suffix) if closedness == 'suffix' else all_prefixes(suffix)
        suffixes.reverse()
        suffix_to_query = suffixes
    else:
        suffix_to_query = [suffix]
    return suffix_to_query


def linear_cex_processing(sul: SUL, cex: tuple, hypothesis, suffix_closedness=True, closedness='suffix',
                          direction='fwd', is_vpa=False):
    assert direction in {'fwd', 'bwd'}

    direction = 'fwd'

    distinguishing_suffix = None
    previous_output = None

    for i in range(0, len(cex)):
        bp = i if direction == 'fwd' else -i - 1
        prefix = cex[:bp]
        suffix = cex[bp:]
        assert cex == prefix + suffix

        hypothesis.reset_to_initial()
        hypothesis.execute_sequence(hypothesis.initial_state, prefix)

        if not is_vpa:
            s_bracket = hypothesis.current_state.prefix
        else:
            s_bracket = tuple(hypothesis.transform_access_string(hypothesis.current_state))

        sul_out = sul.query(s_bracket + suffix)[-1]

        if previous_output is None:
            previous_output = sul_out
            continue

        if sul_out != previous_output:
            distinguishing_suffix = suffix if direction == 'fwd' else cex[bp + 1:]
            break

        previous_output = sul_out

    assert distinguishing_suffix
    if suffix_closedness:
        suffixes = all_suffixes(distinguishing_suffix) if closedness == 'suffix' else all_prefixes(
            distinguishing_suffix)
        suffixes.reverse()
        suffix_to_query = suffixes
    else:
        suffix_to_query = [distinguishing_suffix]

    return suffix_to_query


def exponential_cex_processing(sul: SUL, cex: tuple, hypothesis, suffix_closedness=True, closedness='suffix',
                               direction='fwd', is_vpa=False):
    assert direction in {'fwd', 'bwd'}

    cex_out = sul.query(cex)

    bwd_subtrahend = 1
    if direction == 'fwd':
        bp_recent = 0
        bp = 1
    else:
        bp_recent = len(cex)
        bp = len(cex)-1

    suffix = None
    while True:
        if direction == 'fwd':
            if bp >= len(cex):
                bp = len(cex)
                break
        else:
            if bp <= 1:
                bp = 1
                break

        prefix = cex[:bp]
        suffix = cex[bp:]
        assert cex == prefix + suffix

        hypothesis.reset_to_initial()
        hypothesis.execute_sequence(hypothesis.initial_state, prefix)

        if not is_vpa:
            s_bracket = hypothesis.current_state.prefix
        else:
            s_bracket = tuple(hypothesis.transform_access_string(hypothesis.current_state))

        sul_out = sul.query(s_bracket + suffix)

        if sul_out[-1] != cex_out[-1] and direction == 'fwd':
            break
        elif sul_out[-1] == cex_out[-1] and direction == 'bwd':
            break

        bp_recent = bp
        if direction == 'fwd':
            bp *= 2
        else:
            bp -= bwd_subtrahend
            bwd_subtrahend *= 2

    if (bp - bp_recent) == 1:
        return [suffix]
    else:
        if direction == 'fwd':
            return rs_cex_processing(sul, cex, hypothesis, suffix_closedness, closedness, is_vpa, lower=bp_recent)
        else:
            return rs_cex_processing(sul, cex, hypothesis, suffix_closedness, closedness, is_vpa, upper=bp_recent)


