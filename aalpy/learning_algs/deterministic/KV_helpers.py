from aalpy.automata import Dfa, DfaState


def state_name_gen(prefix='s'):
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1

#
# def pretty_print(hypothesis: Dfa):
#     Dfa(initial_state=hypothesis.initial_state,
#                states=hypothesis.states)

def prettify_hypothesis(hypothesis: Dfa, alphabet, keep_access_strings: bool):
    if not keep_access_strings:
        s = state_name_gen('s')
    old_to_new = {}
    for state in hypothesis.states:
        old_name = state.state_id
        if keep_access_strings:
            new_name = ''.join(str(n) for n in old_name)
        else:
            new_name = next(s)
        old_to_new[old_name] = new_name
        state.state_id = new_name






