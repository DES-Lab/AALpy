from aalpy.automata.Sevpa import Sevpa, SevpaAlphabet


def sevpa_for_L12_refined():
    # Like L12 Language (Balanced parathesis) but the state setup is different

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = ['x']

    input_alphabet = SevpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {")": [("q1", 'pop', ("q0", "("))],
                       "]": [("q1", 'pop', ("q0", "["))],
                       "x": [("q1", None, None)]
                       }),
        "q1": (True, {")": [("q1", 'pop', ("q0", "("))],
                      "]": [("q1", 'pop', ("q0", "["))],
                      "x": [("q0", None, None)]
                      }),
    }
    return Sevpa.from_state_setup(state_setup, "q0", input_alphabet)


def sevpa_congruence_for_vpa_paper():
    # This is a 1-SEVPA which accepts the language L = c1L1r + c2L2r
    # L1 is a regular language which has an even number of a's
    # L2 is a regular language which has an even number of b's

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = ['x']

    input_alphabet = SevpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {")": [("q1", 'pop', ("q0", "("))],
                       "]": [("q1", 'pop', ("q0", "["))],
                       "x": [("q1", None, None)]
                       }),
        "q1": (True, {")": [("q1", 'pop', ("q0", "("))],
                      "]": [("q1", 'pop', ("q0", "["))],
                      "x": [("q0", None, None)]
                      }),
    }
    sevpa = Sevpa.from_state_setup(state_setup, "q0", input_alphabet)
    return sevpa
