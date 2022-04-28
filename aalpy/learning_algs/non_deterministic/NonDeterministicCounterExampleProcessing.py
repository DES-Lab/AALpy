from aalpy.utils.HelperFunctions import all_suffixes


def non_det_longest_prefix_cex_processing(observation_table, cex: tuple):
    """
    Suffix processing strategy found in Shahbaz-Groz paper 'Inferring Mealy Machines'.
    It splits the counterexample into prefix and suffix. Prefix is the longest element of the S union S.A that
    matches the beginning of the counterexample. By removing such prefix from counterexample, no consistency check
    is needed.

    Args:

        observation_table: non-deterministic observation table
        cex: counterexample (inputs/outputs)

    Returns:

        suffixes to add to the E set
    """
    prefixes = observation_table.S + observation_table.get_extended_S()
    prefixes.reverse()
    trimmed_suffix = None

    cex = tuple(cex[0])  # cex[0] are inputs, cex[1] are outputs
    for p in prefixes:
        prefix_inputs = p[0]
        if prefix_inputs == tuple(cex[:len(prefix_inputs)]):
            trimmed_suffix = cex[len(prefix_inputs):]
            break

    if trimmed_suffix:
        suffixes = all_suffixes(trimmed_suffix)
    else:
        suffixes = all_suffixes(cex)
    suffixes.reverse()
    return suffixes


def non_det_rs_cex_processing(observation_table, hypothesis, cex):
    """
    Experimental RS counterexample processing.

    Args:

        observation_table: non-deterministic observation table
        hypothesis: current hypothesis
        cex: counterexample (inputs/outputs)

    Returns:

        suffixes to add to the E set
    """
    cex_out = cex[1]
    cex_input = list(cex[0])

    lower = 1
    upper = len(cex_input) - 2

    while True:
        hypothesis.reset_to_initial()
        mid = (lower + upper) // 2

        # arr[:n] -> first n values
        # arr[n:] -> last n values

        for s_p in cex[:mid]:
            hypothesis.step_to(s_p[0], s_p[1])

        s_bracket = hypothesis.current_state.prefix[0]

        d = tuple(cex_input[mid:])
        target_output_reached = False
        # some random heuristic to determine how much to sample each candidate
        for _ in range(min(len(cex_input) * 2, 50)):
            mq = observation_table.sul.query(s_bracket + d)
            if mq[-1] == cex_out[-1]:
                target_output_reached = True
                break

        if target_output_reached:  # only check if the last element is the same as the cex
            lower = mid + 1
            if upper < lower:
                suffix = tuple(d[1:])
                break
        else:
            upper = mid - 1
            if upper < lower:
                suffix = d
                break
    if suffix == cex_input:
        return non_det_longest_prefix_cex_processing(observation_table, cex)

    return all_suffixes(suffix)
