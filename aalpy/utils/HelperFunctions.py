import string
from collections import defaultdict


def extend_set(list_to_extend: list, new_elements: list) -> list:
    """
    Helper function to extend a list while maintaining set property.
    They are stored as lists, so with this function set property is maintained.
    :return

    Returns:

        list of elements that were added to the set

    """
    set_repr = set(list_to_extend)
    added_elements = [s for s in new_elements if s not in set_repr]
    list_to_extend.extend(added_elements)
    return added_elements


def all_prefixes(li):
    """
    Returns all prefixes of a list.

    Args:
      li: list from which to compute all prefixes

    Returns:
      list of all prefixes

    """
    return [tuple(li[:i + 1]) for i in range(len(li))]


def all_suffixes(li):
    """
    Returns all suffixes of a list.

    Args:
      li: list from which to compute all suffixes

    Returns:
      list of all suffixes

    """
    return [tuple(li[len(li) - i - 1:]) for i in range(len(li))]


def profile_function(function: callable, sort_key='cumtime'):
    """

    Args:
      function: callable: 
      sort_key:  (Default value = 'cumtime')

    Returns:
        prints the profiling results
    """
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    function()
    pr.disable()
    pr.print_stats(sort=sort_key)


def random_string_generator(size=10, chars=string.ascii_lowercase + string.digits):
    """

    Args:

      size:  (Default value = 10)
      chars:  (Default value = string.ascii_lowercase + string.digits)

    Returns:

        a random string of length size
    """
    import random
    return ''.join(random.choice(chars) for _ in range(size))


def print_learning_info(info: dict):
    """
    Print learning statistics.
    """
    print('-----------------------------------')
    print('Learning Finished.')
    print('Learning Rounds:  {}'.format(info['learning_rounds']))
    print('Number of states: {}'.format(info['automaton_size']))
    print('Time (in seconds)')
    print('  Total                : {}'.format(info['total_time']))
    print('  Learning algorithm   : {}'.format(info['learning_time']))
    print('  Conformance checking : {}'.format(info['eq_oracle_time']))
    print('Learning Algorithm')
    print(' # Membership Queries  : {}'.format(info['queries_learning']))
    if 'cache_saved' in info.keys():
        print(' # MQ Saved by Caching : {}'.format(info['cache_saved']))
    print(' # Steps               : {}'.format(info['steps_learning']))
    print('Equivalence Query')
    print(' # Membership Queries  : {}'.format(info['queries_eq_oracle']))
    print(' # Steps               : {}'.format(info['steps_eq_oracle']))
    print('-----------------------------------')


def print_observation_table(ot, table_type):
    """
    Prints the whole observation table.

    Args:

        ot: observation table
        table_type: 'det', 'non-det', or 'stoc'

    """
    if table_type == 'det':
        s_set, extended_s, e_set, table = ot.S, ot.s_dot_a(), ot.E, ot.T
    elif table_type == 'non-det':
        s_set, extended_s, e_set, table = ot.S, ot.S_dot_A, ot.E, ot.T
    else:
        s_set, extended_s, e_set, table = ot.S, ot.get_extended_s(), ot.E, ot.T

    headers = [str(e) for e in e_set]
    s_rows = []
    extended_rows = []
    headers.insert(0, 'Prefixes / E set')
    for s in s_set:
        row = [str(s)]
        if table_type == 'det':
            row.extend(str(e) for e in table[s])
        else:
            row.extend(str(table[s][e]) for e in e_set)
        s_rows.append(row)
    for s in extended_s:
        row = [str(s)]
        if table_type == 'det':
            row.extend(str(e) for e in table[s])
        else:
            row.extend(str(table[s][e]) for e in e_set)
        extended_rows.append(row)

    table = [headers] + s_rows
    columns = defaultdict(int)
    for i in table + extended_rows:
        for index, el in enumerate(i):
            columns[index] = max(columns[index], len(el))

    row_len = 0
    for row in table:
        row = "|".join(element.ljust(columns[ind] + 1) for ind, element in enumerate(row))
        print("-" * len(row))
        row_len = len(row)
        print(row)
    print('=' * row_len)
    for row in extended_rows:
        row = "|".join(element.ljust(columns[ind] + 1) for ind, element in enumerate(row))
        print("-" * len(row))
        print(row)
    print('-' * row_len)


def is_suffix_of(suffix, trace) -> bool:
    """

    Args:
      suffix: target suffix
      trace: trace in question

    Returns:

        True if suffix is the suffix of trace.
    """
    if len(trace) < len(suffix):
        return False
    else:
        return trace[-len(suffix):] == suffix


def get_cex_prefixes(cex, automaton_type):
    """
    Returns all prefixes of the stochastic automaton.

    Args:
        cex: counterexample
        automaton_type: `mdp` or `smm`

    Returns:

        all prefixes of the counterexample based on the `automaton_type`
    """
    if automaton_type == 'mdp':
        return [tuple(cex[:i + 1]) for i in range(0, len(cex), 2)]
    return [tuple(cex[:i]) for i in range(0, len(cex) + 1, 2)]


def get_available_oracles_and_err_msg():
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.oracles import RandomWordEqOracle
    available_oracles = {RandomWalkEqOracle, RandomWordEqOracle}

    available_oracles_msg = 'Warning! Only Random Walk and Random Word oracles are supported for non-deterministic and ' \
                            'stochastic learning. If you have implemented the custom oracle, set the custom_oracle flag ' \
                            'to True. '

    return available_oracles, available_oracles_msg
