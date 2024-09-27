import ast
import random

from aalpy.automata.Vpa import Vpa, VpaAlphabet


def vpa_for_L1():
    # we always ensure that n >= 1

    call_set = ['a']
    return_set = ['b']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "a")], "b": [(Vpa.error_state.state_id, None, None)]}),
        "q1": (False, {"a": [("q1", 'push', "a")], "b": [("q2", 'pop', "a")]}),
        "q2": (True, {"a": [(Vpa.error_state.state_id, None, None)], "b": [("q2", 'pop', "a")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L2():
    call_set = ['a', 'b']
    return_set = ['c', 'd']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L3():
    call_set = ['a', 'c', 'b', 'd']
    return_set = ['e', 'g', 'f', 'h']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L4():
    call_set = ['a', 'b']
    return_set = ['c', 'd']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q01", 'push', "a")], "b": [(Vpa.error_state.state_id, None, None)]}),
        "q01": (False, {"b": [("q1", 'push', "b")], "a": [(Vpa.error_state.state_id, None, None)]}),

        "q1": (False, {"a": [("q11", 'push', "a")], "b": [(Vpa.error_state.state_id, None, None)],
                       "c": [("q21", 'pop', "b")]}),
        "q11": (False, {"b": [("q1", 'push', "b")], "a": [(Vpa.error_state.state_id, None, None)]}),
        "q21": (False, {"d": [("q2", 'pop', "a")]}),
        "q2": (True, {"c": [("q21", 'pop', "b")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L5():
    call_set = ['a', 'b', 'c']
    return_set = ['d', 'e', 'f']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L7():
    # Dyck order 2

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L8():
    # Dyck order 3

    call_set = ['(', '[', '{']
    return_set = [')', ']', '}']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L9():
    # Dyck order 4

    call_set = ['(', '[', '{', '<']
    return_set = [')', ']', '}', '>']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
                      ">": [("q1", 'pop', "<")],
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L10():
    # RE Dyck order 1

    call_set = ['a']
    return_set = ['v']
    internal_set = ['b', 'c', 'd', 'e', 'w', 'x', 'y', 'z']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L11():
    # RE Dyck order 1

    call_set = ['c1', 'c2']
    return_set = ['r1', 'r2']
    internal_set = ['i1', 'i2']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"c1": [("qa", 'push', "c1")],
                       "c2": [("q1", 'push', "c2")],
                       }),
        "qa": (False, {"i1": [("q1", None, None)],
                       }),
        "q1": (True, {"c1": [("qa", 'push', "c1")],
                      "c2": [("q1", 'push', "c2")],
                      "r1": [("qd", 'pop', "c1"), ("qd", 'pop', "c2")],
                      "r2": [("q1", 'pop', "c1"), ("q1", 'pop', "c2")]}),
        "qd": (False, {"i2": [("q1", None, None)]})
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L12():
    # Dyck order 2 (single-nested)

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L13():
    # Dyck order 1

    call_set = ['(']
    return_set = [')']
    internal_set = ['a', 'b', 'c']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L14():
    # Dyck order 2

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = ['a', 'b', 'c']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L15():
    # Dyck order 1

    call_set = ['(']
    return_set = [')']
    internal_set = ['a', 'b', 'c', 'd']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

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
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L16():
    # just a testing language

    call_set = ['a']
    return_set = ['b']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "$")]}),
        "q1": (False, {"a": [("q1", 'push', "x")],
                       "b": [("q1", 'pop', "x"), ("q2", 'pop', "$")],
                       }),
        "q2": (True, {})
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def gen_arithmetic_data(num_sequances=3000, min_seq_len=2, max_seq_len=8):
    from aalpy.base import SUL
    from aalpy.utils import convert_i_o_traces_for_RPNI

    class ArithmeticSUL(SUL):
        def __init__(self):
            super().__init__()
            self.string_under_test = ''

        def pre(self):
            self.string_under_test = ''

        def post(self):
            pass

        def step(self, letter):
            if letter:
                self.string_under_test += ' ' + letter if len(self.string_under_test) > 0 else letter

            try:
                # Parse the expression using ast
                parsed_expr = ast.parse(self.string_under_test, mode='eval')
                # Check if the parsed expression is a valid arithmetic expression
                is_valid = all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Name, ast.Load))
                               or isinstance(node, ast.operator) or isinstance(node, ast.expr_context)
                               or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.operator))
                               for node in ast.walk(parsed_expr))
                return is_valid

            except SyntaxError:
                return False

    sul = ArithmeticSUL()

    alphabet = VpaAlphabet(internal_alphabet=['1', '+', '-', '/', '0'], call_alphabet=['(', ], return_alphabet=[')', ])
    merged_alphabet = alphabet.get_merged_alphabet()
    data = []
    while len(data) < num_sequances:
        seq = []
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            seq.append(random.choice(merged_alphabet))

            outputs = sul.query(tuple(seq))
            data.append(list(zip(seq, outputs)))

    rpni_format = convert_i_o_traces_for_RPNI(data)

    return rpni_format, alphabet


def get_all_VPAs():
    return [vpa_for_L1(), vpa_for_L2(), vpa_for_L4(), vpa_for_L5(), vpa_for_L7(), vpa_for_L8(),
            vpa_for_L9(),
            vpa_for_L10(), vpa_for_L11(), vpa_for_L12(), vpa_for_L13(), vpa_for_L14(), vpa_for_L15()]
