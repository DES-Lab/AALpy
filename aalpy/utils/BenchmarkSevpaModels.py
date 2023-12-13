from aalpy.automata.Sevpa import Sevpa
from aalpy.utils import load_automaton_from_file


def sevpa_for_L1():
    state_setup = {
        'q0': (False, {'b': [('q1', 'pop', ('q0', 'a'))]
                       }),
        'q1': (True, {'b': [('q1', 'pop', ('q0', 'a'))]
                      })
    }
    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L2():
    state_setup = {
        'q0': (False, {'d': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))],
                       'c': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))]
                       }),
        'q1': (True, {'d': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))],
                      'c': [('q1', 'pop', ('q0', 'a')), ('q1', 'pop', ('q0', 'b'))]
                      })
    }

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L3():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L4():
    state_setup = {
        'q0': (False, {'c': [('q2', 'pop', ('q0', 'b'))]
                       }),
        'q1': (True, {'c': [('q2', 'pop', ('q0', 'b'))]
                      }),
        'q2': (False, {'d': [('q1', 'pop', ('q0', 'a'))]
                       })
    }
    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L5():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L7():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L8():
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
    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L9():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L10():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L11():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L12():
    state_setup = {
        'q0': (False, {']': [('q1', 'pop', ('q0', '['))],
                       ')': [('q1', 'pop', ('q0', '('))]
                       }),
        'q1': (True, {']': [('q1', 'pop', ('q0', '['))],
                      ')': [('q1', 'pop', ('q0', '('))]
                      })
    }

    return Sevpa.from_state_setup(state_setup, init_state_id="q0", )


def sevpa_for_L13():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L14():
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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


def sevpa_for_L15():
    # Dyck order 1

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

    return Sevpa.from_state_setup(state_setup, init_state_id="q0")


if __name__ == '__main__':
    e = sevpa_for_L13()
    print(e)
    print(e.get_input_alphabet())
    e.save('test')
    m = load_automaton_from_file('test.dot', automaton_type='vpa')
    print('Loaded')
    print(m)

