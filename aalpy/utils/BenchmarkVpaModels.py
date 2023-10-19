from aalpy.automata.Vpa import Vpa

def vpa_for_L1():
    # we always ensure that n >= 1

    call_set = {'a'}
    return_set = {'b'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "a")], "b": [(Vpa.error_state.state_id, None, None)]}),
        "q1": (False, {"a": [("q1", 'push', "a")], "b": [("q2", 'pop', "a")]}),
        "q2": (True, {"a": [(Vpa.error_state.state_id, None, None)], "b": [("q2", 'pop', "a")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L2():

    call_set = {'a', 'b'}
    return_set = {'c', 'd'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "a")], "b": [("q1", 'push', "b")],
                       "c": [(Vpa.error_state.state_id, None, None)],
                       "d": [(Vpa.error_state.state_id, None, None)]}),
        "q1": (False, {"a": [("q1", 'push', "a")], "b": [("q1", 'push', "b")],
                       "c": [("q2", 'pop', "a"), ("q2", 'pop', "b")],
                       "d": [("q2", 'pop', "a"), ("q2", 'pop', "b")]}),
        "q2": (True, {"a": [(Vpa.error_state.state_id, None, None)],
                      "b": [(Vpa.error_state.state_id, None, None)],
                      "c": [("q2", 'pop', "a"), ("q2", 'pop', "b")],
                      "d": [("q2", 'pop', "a"), ("q2", 'pop', "b")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L3():

    call_set = {'a', 'c', 'b', 'd'}
    return_set = {'e', 'g', 'f', 'h'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"a": [("q0a", 'push', "a")],
                       "c": [("q0c", 'push', "c")],
                       }),
        "q0a": (False, {"b": [("q1", 'push', "b")]}),
        "q0c": (False, {"d": [("q1", 'push', "d")]}),
        "q1": (False, {"a": [("q1a", 'push', "a")],
                       "c": [("q1c", 'push', "c")],
                       "e": [("q2e", 'pop', "b"), ("q2e", 'pop', "d")],
                       "g": [("q2g", 'pop', "b"), ("q2g", 'pop', "d")],  # stack should actually be redundant
                       }),
        "q1a": (False, {"b": [("q1", 'push', "b")]}),
        "q1c": (False, {"d": [("q1", 'push', "d")]}),
        "q2e": (False, {"f": [("q2", 'pop', "a"), ("q2", 'pop', "c")]}),
        "q2g": (False, {"h": [("q2", 'pop', "a"), ("q2", 'pop', "c")]}),
        "q2": (True, {"e": [("q2e", 'pop', "b"), ("q2e", 'pop', "d")],
                      "g": [("q2g", 'pop', "b"), ("q2g", 'pop', "d")]})
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L4():

    call_set = {'a', 'b'}
    return_set = {'c', 'd'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"a": [("q01", 'push', "a")], "b": [(Vpa.error_state.state_id, None, None)]}),
        "q01": (False, {"b": [("q1", 'push', "b")], "a": [(Vpa.error_state.state_id, None, None)]}),

        "q1": (False, {"a": [("q11", 'push', "a")], "b": [(Vpa.error_state.state_id, None, None)],
                       "c": [("q21", 'pop', "b")]}),
        "q11": (False, {"b": [("q1", 'push', "b")], "a": [(Vpa.error_state.state_id, None, None)]}),
        "q21": (False, {"d": [("q2", 'pop', "a")]}),
        "q2": (True, {"c": [("q21", 'pop', "b")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L5():

    call_set = {'a', 'b', 'c'}
    return_set = {'d', 'e', 'f'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"a": [("q01", 'push', "a")]}),
        "q01": (False, {"b": [("q02", 'push', "b")]}),
        "q02": (False, {"c": [("q1", 'push', "c")]}),
        "q1": (False, {"a": [("q11", 'push', "a")],
                       "d": [("q21", 'pop', "c")]}),
        "q11": (False, {"b": [("q12", 'push', "b")]}),
        "q12": (False, {"c": [("q1", 'push', "c")]}),
        "q21": (False, {"e": [("q22", 'pop', "b")]}),
        "q22": (False, {"f": [("q2", 'pop', "a")]}),
        "q2": (True, {"d": [("q21", 'pop', "c")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L7():
    # Dyck order 2

    call_set = {'(', '['}
    return_set = {')', ']'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', '(')],
                       "[": [("q1", 'push', '[')],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', '(')],
                      "[": [("q1", 'push', '[')],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")]
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L8():
    # Dyck order 3

    call_set = {'(', '[', '{'}
    return_set = {')', ']', '}'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', '(')],
                       "[": [("q1", 'push', '[')],
                       "{": [("q1", 'push', '{')],
                       }),
        "q1": (True, {"(": [("q1", 'push', '(')],
                      "[": [("q1", 'push', '[')],
                      "{": [("q1", 'push', '{')],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "}": [("q1", 'pop', "{")],
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L9():
    # Dyck order 4

    call_set = {'(', '[', '{', '<'}
    return_set = {')', ']', '}', '>'}
    internal_set = {}

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', '(')],
                       "[": [("q1", 'push', '[')],
                       "{": [("q1", 'push', '{')],
                       "<": [("q1", 'push', '<')],
                       }),
        "q1": (True, {"(": [("q1", 'push', '(')],
                      "[": [("q1", 'push', '[')],
                      "{": [("q1", 'push', '{')],
                      "<": [("q1", 'push', '<')],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "}": [("q1", 'pop', "{")],
                      ">": [("q1", 'pop', ">")],
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L10():
    # RE Dyck order 1

    call_set = {'a'}
    return_set = {'v'}
    internal_set = {'b', 'c', 'd', ' e', 'w', 'x', 'y', 'z'}

    state_setup = {
        "q0": (False, {"a": [("qa", 'push', "a")],
                       }),
        "qa": (False, {"b": [("qb", None, None)],
                       }),
        "qb": (False, {"c": [("qc", None, None)],
                       }),
        "qc": (False, {"d": [("qd", None, None)],
                       }),
        "qd": (False, {"e": [("q1", None, None)],
                       }),
        "q1": (True, {"a": [("qa", 'push', "a")],
                      "v": [("qv", 'pop', "a")]}),
        "qv": (False, {"w": [("qw", None, None)]}),
        "qw": (False, {"x": [("qx", None, None)]}),
        "qx": (False, {"y": [("qy", None, None)]}),
        "qy": (False, {"z": [("q1", None, None)]})
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L11():
    # RE Dyck order 1

    call_set = {'a', 'c'}
    return_set = {'d', 'f'}
    internal_set = {'b', 'e'}

    state_setup = {
        "q0": (False, {"a": [("qa", 'push', "a")],
                       "c": [("q1", 'push', "c")],
                       }),
        "qa": (False, {"b": [("q1", None, None)],
                       }),
        "q1": (True, {"a": [("qa", 'push', "a")],
                      "c": [("q1", 'push', "c")],
                      "d": [("qd", 'pop', "a"), ("qd", 'pop', "c")],
                      "f": [("q1", 'pop', "a"), ("q1", 'pop', "c")]}),
        "qd": (False, {"e": [("q1", None, None)]})
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L12():
    # Dyck order 2 (single-nested)

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = []

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "[": [("q1", 'push', "[")],  # exclude empty seq
                       }),
        "q1": (False, {"(": [("q1", 'push', "(")],
                       "[": [("q1", 'push', "[")],
                       ")": [("q2", 'pop', "(")],
                       "]": [("q2", 'pop', "[")]}),
        "q2": (True, {
            ")": [("q2", 'pop', "(")],
            "]": [("q2", 'pop', "[")]
        }),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L13():
    # Dyck order 1

    call_set = {'('}
    return_set = {')'}
    internal_set = {'a', 'b', 'c'}

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "a": [("q1", None, None)],
                       "b": [("q1", None, None)],
                       "c": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', "(")],
                      ")": [("q1", 'pop', "(")],
                      "a": [("q1", None, None)],
                      "b": [("q1", None, None)],
                      "c": [("q1", None, None)]
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L14():
    # Dyck order 2

    call_set = {'(', '['}
    return_set = {')', ']'}
    internal_set = {'a', 'b', 'c'}

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "[": [("q1", 'push', "[")],
                       "a": [("q1", None, None)],
                       "b": [("q1", None, None)],
                       "c": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', "(")],
                      "[": [("q1", 'push', "[")],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "a": [("q1", None, None)],
                      "b": [("q1", None, None)],
                      "c": [("q1", None, None)]
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L15():
    # Dyck order 1

    call_set = {'('}
    return_set = {')'}
    internal_set = {'a', 'b', 'c', 'd'}

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "a": [("qa", None, None)],
                       "d": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', "(")],
                      ")": [("q1", 'pop', "(")],
                      "a": [("qa", None, None)],
                      "d": [("q1", None, None)],
                      }),
        "qa": (False, {"b": [("qb", None, None)],
                       }),
        "qb": (False, {"c": [("q1", None, None)],
                       })
    }
    vpa = Vpa.from_state_setup(state_setup, "q0", call_set, return_set, internal_set)
    return vpa


def vpa_for_L16():
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



