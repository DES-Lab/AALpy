import string
from collections import defaultdict

from aalpy.automata import Mdp, StochasticMealyMachine, MdpState


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


def smm_to_mdp_conversion(smm: StochasticMealyMachine):
    """
    Convert SMM to MDP.

    Args:
      smm: StochasticMealyMachine: SMM to convert

    Returns:

        equivalent MDP

    """
    inputs = smm.get_input_alphabet()
    mdp_states = []
    smm_state_to_mdp_state = dict()
    init_state = MdpState("0", "___start___")
    mdp_states.append(init_state)
    for s in smm.states:
        incoming_edges = defaultdict(list)
        incoming_outputs = set()
        for pre_s in smm.states:
            for i in inputs:
                incoming_edges[i] += filter(lambda t: t[0] == s, pre_s.transitions[i])
                incoming_outputs.update(map(lambda t: t[1], incoming_edges[i]))
        state_id = 0
        for o in incoming_outputs:
            new_state_id = s.state_id + str(state_id)
            state_id += 1
            new_state = MdpState(new_state_id, o)
            mdp_states.append(new_state)
            smm_state_to_mdp_state[(s.state_id, o)] = new_state

    for s in smm.states:
        mdp_states_for_s = {mdp_state for (s_id, o), mdp_state in smm_state_to_mdp_state.items() if s_id == s.state_id}
        for i in inputs:
            for outgoing_t in s.transitions[i]:
                target_smm_state = outgoing_t[0]
                output = outgoing_t[1]
                prob = outgoing_t[2]
                target_mdp_state = smm_state_to_mdp_state[(target_smm_state.state_id, output)]
                for mdp_state in mdp_states_for_s:
                    mdp_state.transitions[i].append((target_mdp_state, prob))
                if s == smm.initial_state:
                    init_state.transitions[i].append((target_mdp_state, prob))
    return Mdp(init_state, mdp_states)


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
