from aalpy.automata.Sevpa import Sevpa, SevpaAlphabet


def sevpa_for_L1():
    call_set = ['a']
    return_set = ['b']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'b': [('q1', 'pop', ('q0', 'a'))]
                       }),
        'q1': (True, {'b': [('q1', 'pop', ('q0', 'a'))]
                      })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L2():
    call_set = ['a', 'b']
    return_set = ['c', 'd']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'d': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))],
                       'c': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))]
                       }),
        'q1': (True, {'d': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))],
                      'c': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))]
                      })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L3():
    call_set = ['a', 'c', 'b', 'd']
    return_set = ['e', 'g', 'f', 'h']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'g': [('q6', 'pop', ('q0', 'd')),
                             ('q4', 'pop', ('q0', 'b'))],
                       'e': [('q5', 'pop', ('q0', 'd')),
                             ('q2', 'pop', ('q0', 'b'))]
                       }),
        'q1': (True, {'g': [('q6', 'pop', ('q0', 'd')),
                            ('q4', 'pop', ('q0', 'b'))],
                      'e': [('q5', 'pop', ('q0', 'd')),
                            ('q2', 'pop', ('q0', 'b'))]
                      }),
        'q2': (False, {'f': [('q1', 'pop', ('q0', 'a'))]
                       }),
        'q4': (False, {'h': [('q1', 'pop', ('q0', 'a'))]
                       }),
        'q5': (False, {'f': [('q1', 'pop', ('q0', 'c'))]
                       }),
        'q6': (False, {'h': [('q1', 'pop', ('q0', 'c'))]
                       })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L4():
    call_set = ['a', 'b']
    return_set = ['c', 'd']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'c': [('q2', 'pop', ('q0', 'b'))]
                       }),
        'q1': (True, {'c': [('q2', 'pop', ('q0', 'b'))]
                      }),
        'q2': (False, {'d': [('q1', 'pop', ('q0', 'a'))]
                       })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L5():
    call_set = ['a', 'b', 'c']
    return_set = ['d', 'e', 'f']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'d': [('q2', 'pop', ('q0', 'c'))]
                       }),
        'q1': (True, {'d': [('q2', 'pop', ('q0', 'c'))]
                      }),
        'q2': (False, {'e': [('q3', 'pop', ('q0', 'b'))]
                       }),
        'q3': (False, {'f': [('q1', 'pop', ('q0', 'a'))]
                       })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L7():
    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {')': [('q1', 'pop', ('q0', '(')),
                             ('q1', 'pop', ('q1', '('))],
                       ']': [('q1', 'pop', ('q0', '[')),
                             ('q1', 'pop', ('q1', '['))]
                       }),
        'q1': (True, {')': [('q1', 'pop', ('q0', '(')),
                            ('q1', 'pop', ('q1', '('))],
                      ']': [('q1', 'pop', ('q0', '[')),
                            ('q1', 'pop', ('q1', '['))]
                      })

    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L8():
    call_set = ['(', '[', '{']
    return_set = [')', ']', '}']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {')': [('q1', 'pop', ('q0', '(')),
                             ('q1', 'pop', ('q1', '('))],
                       '}': [('q1', 'pop', ('q0', '{')),
                             ('q1', 'pop', ('q1', '{'))],
                       ']': [('q1', 'pop', ('q0', '[')),
                             ('q1', 'pop', ('q1', '['))]
                       }),
        'q1': (True, {')': [('q1', 'pop', ('q0', '(')),
                            ('q1', 'pop', ('q1', '('))],
                      '}': [('q1', 'pop', ('q0', '{')),
                            ('q1', 'pop', ('q1', '{'))],
                      ']': [('q1', 'pop', ('q0', '[')),
                            ('q1', 'pop', ('q1', '['))]
                      })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L9():
    call_set = ['(', '[', '{', '<']
    return_set = [')', ']', '}', '>']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {']': [('q1', 'pop', ('q0', '[')),
                             ('q1', 'pop', ('q1', '['))],
                       '}': [('q1', 'pop', ('q0', '{')),
                             ('q1', 'pop', ('q1', '{'))],
                       ')': [('q1', 'pop', ('q0', '(')),
                             ('q1', 'pop', ('q1', '('))],
                       '>': [('q1', 'pop', ('q0', '<')),
                             ('q1', 'pop', ('q1', '<'))]
                       }),
        'q1': (True, {']': [('q1', 'pop', ('q0', '[')),
                            ('q1', 'pop', ('q1', '['))],
                      '}': [('q1', 'pop', ('q0', '{')),
                            ('q1', 'pop', ('q1', '{'))],
                      ')': [('q1', 'pop', ('q0', '(')),
                            ('q1', 'pop', ('q1', '('))],
                      '>': [('q1', 'pop', ('q0', '<')),
                            ('q1', 'pop', ('q1', '<'))]
                      })
    }

    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L10():
    call_set = ['a']
    return_set = ['v']
    internal_set = ['b', 'c', 'd', 'e', 'w', 'x', 'y', 'z']

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        "q0": (False, {"b": [("qb", None, None)],
                       }),
        "qb": (False, {"c": [("qc", None, None)],
                       }),
        "qc": (False, {"d": [("qd", None, None)],
                       }),
        "qd": (False, {"e": [("q1", None, None)],
                       }),
        "q1": (False, {"v": [("qv", 'pop', ('q0', 'a')),
                             ("qv", 'pop', ('q1', 'a')),
                             ("qv", 'pop', ('q2', 'a'))]
                       }),
        "qv": (False, {"w": [("qw", None, None)]
                       }),
        "qw": (False, {"x": [("qx", None, None)]
                       }),
        "qx": (False, {"y": [("qy", None, None)]
                       }),
        "qy": (False, {"z": [("q2", None, None)]
                       }),
        "q2": (True, {"v": [("qv", 'pop', ('q0', 'a')),
                            ("qv", 'pop', ('q1', 'a')),
                            ("qv", 'pop', ('q2', 'a'))]
                      })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L11():
    call_set = ['c1', 'c2']
    return_set = ['r1', 'r2']
    internal_set = ['i1', 'i2']

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'i1': [('q2', None, None)],
                       'r1': [('q3', 'pop', ('q0', 'c2')),
                              ('q3', 'pop', ('q1', 'c2')),
                              ('q5', 'pop', ('q2', 'c2'))],
                       'r2': [('q1', 'pop', ('q0', 'c2')),
                              ('q1', 'pop', ('q1', 'c2')),
                              ('q2', 'pop', ('q2', 'c2'))]
                       }),
        'q1': (True, {'r1': [('q3', 'pop', ('q0', 'c2')),
                             ('q3', 'pop', ('q1', 'c2')),
                             ('q5', 'pop', ('q2', 'c2'))],
                      'r2': [('q1', 'pop', ('q0', 'c2')),
                             ('q1', 'pop', ('q1', 'c2')),
                             ('q2', 'pop', ('q2', 'c2'))]
                      }),
        'q2': (False, {'r1': [('q3', 'pop', ('q0', 'c1')),
                              ('q3', 'pop', ('q1', 'c1')),
                              ('q5', 'pop', ('q2', 'c1'))],
                       'r2': [('q1', 'pop', ('q0', 'c1')),
                              ('q1', 'pop', ('q1', 'c1')),
                              ('q2', 'pop', ('q2', 'c1'))]
                       }),
        'q3': (False, {'i2': [('q1', None, None)]
                       }),
        'q5': (False, {'i2': [('q2', None, None)]
                       })
    }

    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L12():
    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = []

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {']': [('q1', 'pop', ('q0', '['))],
                       ')': [('q1', 'pop', ('q0', '('))]
                       }),
        'q1': (True, {']': [('q1', 'pop', ('q0', '['))],
                      ')': [('q1', 'pop', ('q0', '('))]
                      })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L13():
    call_set = ['(']
    return_set = [')']
    internal_set = ['a', 'b', 'c']

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'c': [('q1', None, None)],
                       'b': [('q1', None, None)],
                       'a': [('q1', None, None)],
                       ')': [('q1', 'pop', ('q0', '(')),
                             ('q1', 'pop', ('q1', '('))]
                       }),
        'q1': (True, {'c': [('q1', None, None)],
                      'b': [('q1', None, None)],
                      'a': [('q1', None, None)],
                      ')': [('q1', 'pop', ('q0', '(')),
                            ('q1', 'pop', ('q1', '('))]
                      })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L14():
    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = ['a', 'b', 'c']

    input_alphabet = SevpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    state_setup = {
        'q0': (False, {'a': [('q1', None, None)],
                       'b': [('q1', None, None)],
                       'c': [('q1', None, None)],
                       ']': [('q1', 'pop', ('q0', '[')),
                             ('q1', 'pop', ('q1', '['))],
                       ')': [('q1', 'pop', ('q0', '(')),
                             ('q1', 'pop', ('q1', '('))]
                       }),
        'q1': (True, {'a': [('q1', None, None)],
                      'b': [('q1', None, None)],
                      'c': [('q1', None, None)],
                      ']': [('q1', 'pop', ('q0', '[')),
                            ('q1', 'pop', ('q1', '['))],
                      ')': [('q1', 'pop', ('q0', '(')),
                            ('q1', 'pop', ('q1', '('))]
                      })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa


def sevpa_for_L15():
    # Dyck order 1

    call_set = ['(']
    return_set = [')']
    internal_set = ['a', 'b', 'c', 'd']

    input_alphabet = SevpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        'q0': (False, {'d': [('q1', None, None)],
                       'a': [('q2', None, None)],
                       ')': [('q1', 'pop', ('q0', '(')),
                             ('q1', 'pop', ('q1', '('))]
                       }),
        'q1': (True, {'d': [('q1', None, None)],
                      'a': [('q2', None, None)],
                      ')': [('q1', 'pop', ('q0', '(')),
                            ('q1', 'pop', ('q1', '('))]
                      }),
        'q2': (False, {'b': [('q3', None, None)]
                       }),
        'q3': (False, {'c': [('q1', None, None)]
                       })
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa
