from aalpy.automata.Vpa import Vpa

def vpa_for_L1():
    # just a testing language
    call_set = {'a'}
    return_set = {'b'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "$")]}),
        "q1": (False, {"a": [("q1", 'push', "x")],
                       "b": [("q1", 'pop', "x"), ("q2", 'pop', "$")],
                       }),
        "q2": (True, {})
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa

def vpa_for_L13():
    # Dyck order 1

    call_set = {'('}
    return_set = {')'}
    internal_set = {'a', 'b', 'c'}

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', None)],
                       "a": [("q1", None, None)],
                       "b": [("q1", None, None)],
                       "c": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', None)],
                      ")": [("q1", 'pop', "(")],
                      "a": [("q1", None, None)],
                      "b": [("q1", None, None)],
                      "c": [("q1", None, None)]
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa

