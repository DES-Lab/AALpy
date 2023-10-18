from aalpy.automata.Pda import Pda


def pda_for_L1():
    # we always ensure that n >= 1
    state_setup = {
        "q0": (False, {"a": [("q1", 'push', None)], "b": [(Pda.error_state.state_id, None, None)]}),
        "q1": (False, {"a": [("q1", 'push', None)], "b": [("q2", 'pop', "a")]}),
        "q2": (True, {"a": [(Pda.error_state.state_id, None, None)], "b": [("q2", 'pop', "a")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L2():
    state_setup = {
        "q0": (False, {"a": [("q1", 'push', None)], "b": [("q1", 'push', None)],
                       "c": [(Pda.error_state.state_id, None, None)],
                       "d": [(Pda.error_state.state_id, None, None)]}),
        "q1": (False, {"a": [("q1", 'push', None)], "b": [("q1", 'push', None)],
                       "c": [("q2", 'pop', "a"), ("q2", 'pop', "b")],
                       "d": [("q2", 'pop', "a"), ("q2", 'pop', "b")]}),
        "q2": (True, {"a": [(Pda.error_state.state_id, None, None)],
                      "b": [(Pda.error_state.state_id, None, None)],
                      "c": [("q2", 'pop', "a"), ("q2", 'pop', "b")],
                      "d": [("q2", 'pop', "a"), ("q2", 'pop', "b")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L3():
    state_setup = {
        "q0": (False, {"a": [("q0a", 'push', None)],
                       "c": [("q0c", 'push', None)],
                       }),
        "q0a": (False, {"b": [("q1", 'push', None)]}),
        "q0c": (False, {"d": [("q1", 'push', None)]}),
        "q1": (False, {"a": [("q1a", 'push', None)],
                       "c": [("q1c", 'push', None)],
                       "e": [("q2e", 'pop', "b"), ("q2e", 'pop', "d")],
                       "g": [("q2g", 'pop', "b"), ("q2g", 'pop', "d")],  # stack should actually be redundant
                       }),
        "q1a": (False, {"b": [("q1", 'push', None)]}),
        "q1c": (False, {"d": [("q1", 'push', None)]}),
        "q2e": (False, {"f": [("q2", 'pop', "a"), ("q2", 'pop', "c")]}),
        "q2g": (False, {"h": [("q2", 'pop', "a"), ("q2", 'pop', "c")]}),
        "q2": (True, {"e": [("q2e", 'pop', "b"), ("q2e", 'pop', "d")],
                      "g": [("q2g", 'pop', "b"), ("q2g", 'pop', "d")]})
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L4():
    state_setup = {
        "q0": (False, {"a": [("q01", 'push', None)], "b": [(Pda.error_state.state_id, None, None)]}),
        "q01": (False, {"b": [("q1", 'push', None)], "a": [(Pda.error_state.state_id, None, None)]}),

        "q1": (False, {"a": [("q11", 'push', None)], "b": [(Pda.error_state.state_id, None, None)],
                       "c": [("q21", 'pop', "b")]}),
        "q11": (False, {"b": [("q1", 'push', None)], "a": [(Pda.error_state.state_id, None, None)]}),
        "q21": (False, {"d": [("q2", 'pop', "a")]}),
        "q2": (True, {"c": [("q21", 'pop', "b")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L5():
    state_setup = {
        "q0": (False, {"a": [("q01", 'push', None)]}),
        "q01": (False, {"b": [("q02", 'push', None)]}),
        "q02": (False, {"c": [("q1", 'push', None)]}),
        "q1": (False, {"a": [("q11", 'push', None)],
                       "d": [("q21", 'pop', "c")]}),
        "q11": (False, {"b": [("q12", 'push', None)]}),
        "q12": (False, {"c": [("q1", 'push', None)]}),
        "q21": (False, {"e": [("q22", 'pop', "b")]}),
        "q22": (False, {"f": [("q2", 'pop', "a")]}),
        "q2": (True, {"d": [("q21", 'pop', "c")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L7():
    # Dyck order 2
    state_setup = {
        "q0": (False, {"(": [("q1", 'push', None)],
                       "[": [("q1", 'push', None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', None)],
                      "[": [("q1", 'push', None)],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")]
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L8():
    # Dyck order 3
    state_setup = {
        "q0": (False, {"(": [("q1", 'push', None)],
                       "[": [("q1", 'push', None)],
                       "{": [("q1", 'push', None)],
                       }),
        "q1": (True, {"(": [("q1", 'push', None)],
                      "[": [("q1", 'push', None)],
                      "{": [("q1", 'push', None)],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "}": [("q1", 'pop', "{")],
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L9():
    # Dyck order 4
    state_setup = {
        "q0": (False, {"(": [("q1", 'push', None)],
                       "[": [("q1", 'push', None)],
                       "{": [("q1", 'push', None)],
                       "<": [("q1", 'push', None)],
                       }),
        "q1": (True, {"(": [("q1", 'push', None)],
                      "[": [("q1", 'push', None)],
                      "{": [("q1", 'push', None)],
                      "<": [("q1", 'push', None)],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "}": [("q1", 'pop', "{")],
                      ">": [("q1", 'pop', ">")],
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L10():
    # RE Dyck order 1
    state_setup = {
        "q0": (False, {"a": [("qa", 'push', None)],
                       }),
        "qa": (False, {"b": [("qb", None, None)],
                       }),
        "qb": (False, {"c": [("qc", None, None)],
                       }),
        "qc": (False, {"d": [("qd", None, None)],
                       }),
        "qd": (False, {"e": [("q1", None, None)],
                       }),
        "q1": (True, {"a": [("qa", 'push', None)],
                      "v": [("qv", 'pop', "a")]}),
        "qv": (False, {"w": [("qw", None, None)]}),
        "qw": (False, {"x": [("qx", None, None)]}),
        "qx": (False, {"y": [("qy", None, None)]}),
        "qy": (False, {"z": [("q1", None, None)]})
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L11():
    # RE Dyck order 1
    state_setup = {
        "q0": (False, {"a": [("qa", 'push', None)],
                       "c": [("q1", 'push', None)],
                       }),
        "qa": (False, {"b": [("q1", None, None)],
                       }),
        "q1": (True, {"a": [("qa", 'push', None)],
                      "c": [("q1", 'push', None)],
                      "d": [("qd", 'pop', "a"), ("qd", 'pop', "c")],
                      "f": [("q1", 'pop', "a"), ("q1", 'pop', "c")]}),
        "qd": (False, {"e": [("q1", None, None)]})
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L12():
    # Dyck order 2 (single-nested)
    state_setup = {
        "q0": (False, {"(": [("q1", 'push', None)],
                       "[": [("q1", 'push', None)],  # exclude empty seq
                       }),
        "q1": (False, {"(": [("q1", 'push', None)],
                       "[": [("q1", 'push', None)],
                       ")": [("q2", 'pop', "(")],
                       "]": [("q2", 'pop', "[")]}),
        "q2": (True, {
            ")": [("q2", 'pop', "(")],
            "]": [("q2", 'pop', "[")]
        }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L13():
    # Dyck order 1
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
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L14():
    # Dyck order 2
    state_setup = {
        "q0": (False, {"(": [("q1", 'push', None)],
                       "[": [("q1", 'push', None)],
                       "a": [("q1", None, None)],
                       "b": [("q1", None, None)],
                       "c": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', None)],
                      "[": [("q1", 'push', None)],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "a": [("q1", None, None)],
                      "b": [("q1", None, None)],
                      "c": [("q1", None, None)]
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L15():
    # Dyck order 1
    state_setup = {
        "q0": (False, {"(": [("q1", 'push', None)],
                       "a": [("qa", None, None)],
                       "d": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', None)],
                      ")": [("q1", 'pop', "(")],
                      "a": [("qa", None, None)],
                      "d": [("q1", None, None)],
                      }),
        "qa": (False, {"b": [("qb", None, None)],
                       }),
        "qb": (False, {"c": [("q1", None, None)],
                       })
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda
