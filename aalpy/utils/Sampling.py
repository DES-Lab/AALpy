from random import randint, choices, random

from aalpy import MooreMachine, Dfa, NDMooreMachine, Mdp, MarkovChain
from aalpy.base import Automaton, DeterministicAutomaton


def sample_with_length_limits(alphabet, nr_samples, min_len, max_len):
    return [choices(alphabet, k = randint(min_len, max_len)) for _ in range(nr_samples)]


def sample_with_term_prob(alphabet, nr_samples, term_prob):
    ret = []
    for _ in range(nr_samples):
        k = 0
        while term_prob < random():
            k += 1
        ret.append(choices(alphabet, k=k))
    return ret


def get_complete_sample(automaton: DeterministicAutomaton):
    alphabet = automaton.get_input_alphabet()
    automaton.compute_prefixes()
    char_set = automaton.compute_characterization_set()
    infixes = [(x,) for x in alphabet] + [tuple()]
    return [state.prefix + infix + suffix for state in automaton.states for suffix in char_set for infix in infixes]


def get_io_traces(automaton: Automaton, input_traces: list) -> list:
    moore_automata = (MooreMachine, NDMooreMachine, Mdp, MarkovChain)
    is_moore = isinstance(automaton, moore_automata)

    traces = []
    for input_trace in input_traces:
        output_trace = automaton.execute_sequence(automaton.initial_state, input_trace)
        trace = list(zip(input_trace, output_trace))
        if is_moore:
            trace = [automaton.initial_state.output] + trace
        traces.append(trace)
    return traces
