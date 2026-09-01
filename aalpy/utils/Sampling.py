# Utilities for sampling and building input/output datasets from automata.
from collections.abc import Callable
from functools import wraps
from random import randint, choices, random
from typing import Any

from aalpy import MooreMachine, Dfa, NDMooreMachine, Mdp, MarkovChain
from aalpy.base import Automaton, DeterministicAutomaton


def get_io_traces(automaton: Automaton, input_traces: list) -> list:
    """
    Computes input/output traces for a list of input sequences executed on an automaton.

    :param Automaton automaton: Automaton on which the input traces are executed.
    :param list input_traces: List of input sequences.
    :return list: List of traces, where each trace is a list of (input, output) pairs, prefixed with the initial
        output for Moore-like automata.
    """
    moore_automata = (MooreMachine, Dfa, NDMooreMachine, Mdp, MarkovChain)
    is_moore = isinstance(automaton, moore_automata)

    traces = []
    for input_trace in input_traces:
        output_trace = automaton.execute_sequence(automaton.initial_state, input_trace) if input_trace else []

        trace = list(zip(input_trace, output_trace))
        if is_moore:
            trace = [automaton.initial_state.output] + trace
        traces.append(trace)
    return traces


def get_labeled_sequences(automaton: Automaton, input_traces: list) -> list:
    """
    Computes the final output label for a list of input sequences executed on an automaton.

    :param Automaton automaton: Automaton on which the input traces are executed.
    :param list input_traces: List of input sequences.
    :return list: List of (input_sequence, output) pairs, where output is the label reached after the sequence.
    """
    moore_automata = (MooreMachine, Dfa, NDMooreMachine, Mdp, MarkovChain)
    is_moore = isinstance(automaton, moore_automata)

    data = []
    for input_trace in input_traces:
        if len(input_trace) == 0:
            if not is_moore:
                raise ValueError("tried to get label of empty sequence for Mealy automaton.")
            output = automaton.initial_state.output
        else:
            output = automaton.execute_sequence(automaton.initial_state, input_trace)[-1]
        data.append((input_trace, output))
    return data


def get_data_from_input_sequence(automaton: Automaton, input_sequence: list, data_format: str = "io_sequences") -> list:
    """
    Converts a list of input sequences to the requested data format.

    :param Automaton automaton: Automaton on which the input sequences are executed.
    :param list input_sequence: List of input sequences.
    :param str data_format: Either 'io_sequences' or 'labeled_sequences'.
    :return list: The dataset in the requested format.
    """
    if data_format == "io_sequences":
        return get_io_traces(automaton, input_sequence)
    elif data_format == "labeled_sequences":
        return get_labeled_sequences(automaton, input_sequence)
    else:
        raise ValueError(f"invalid data_format {data_format}. must be 'io_sequences' or 'labeled_sequences'")


def support_automaton_arg(require_transform: bool) -> Callable:
    """
    Creates a decorator that allows a sampling function's first argument to be either an alphabet or an automaton,
    and adds an `include_outputs` keyword argument that returns input/output traces instead of bare input sequences.

    :param bool require_transform: If true, an automaton passed as first argument is transformed into its input
        alphabet before being passed to the wrapped function.
    :return Callable: The decorator.
    """
    def decorator(f: Callable) -> Callable:
        """
        Wraps a sampling function to support an automaton as its first argument.

        :param Callable f: Sampling function to wrap.
        :return Callable: The wrapped function.
        """
        @wraps(f)
        def inner(alphabet: Any, *args: Any, include_outputs: bool = False, **kwargs: Any) -> Any:
            """
            Calls the wrapped sampling function, optionally converting the result to input/output traces.

            :param Any alphabet: Input alphabet, or an automaton from which the alphabet is derived.
            :param Any args: Positional arguments forwarded to the wrapped function.
            :param bool include_outputs: If true, returns input/output traces computed on the given automaton.
            :param Any kwargs: Keyword arguments forwarded to the wrapped function.
            :return Any: The sampled data, optionally converted to input/output traces.
            """
            automaton = None
            if isinstance(alphabet, Automaton):
                automaton = alphabet
                if require_transform:
                    alphabet = alphabet.get_input_alphabet()
            traces = f(alphabet, *args, **kwargs)
            if include_outputs:
                if automaton is None:
                    raise ValueError("automaton must be provided")
                traces = get_io_traces(automaton, traces)
            return traces
        return inner
    return decorator


@support_automaton_arg(True)
def sample_with_length_limits(alphabet: list, nr_samples: int, min_len: int, max_len: int) -> list:
    """
    Samples random input sequences with lengths uniformly chosen between given limits.

    :param list alphabet: Input alphabet to sample from.
    :param int nr_samples: Number of sequences to sample.
    :param int min_len: Minimum sequence length.
    :param int max_len: Maximum sequence length.
    :return list: List of sampled input sequences.
    """
    return [choices(alphabet, k = randint(min_len, max_len)) for _ in range(nr_samples)]


@support_automaton_arg(True)
def sample_with_term_prob(alphabet: list, nr_samples: int, term_prob: float) -> list:
    """
    Samples random input sequences whose length is determined by a per-step termination probability.

    :param list alphabet: Input alphabet to sample from.
    :param int nr_samples: Number of sequences to sample.
    :param float term_prob: Probability of terminating the sequence at each step.
    :return list: List of sampled input sequences.
    """
    ret = []
    for _ in range(nr_samples):
        k = 0
        while term_prob < random():
            k += 1
        ret.append(choices(alphabet, k=k))
    return ret


@support_automaton_arg(False)
def get_complete_sample(automaton: DeterministicAutomaton) -> list:
    """
    Generates a complete sample of an automaton, combining state prefixes, single-input infixes and the
    characterization set suffixes.

    :param DeterministicAutomaton automaton: Automaton for which the complete sample is generated.
    :return list: List of input sequences forming the complete sample.
    """
    alphabet = automaton.get_input_alphabet()
    automaton.compute_prefixes()
    char_set = automaton.compute_characterization_set()
    infixes = [(x,) for x in alphabet] + [tuple()]
    return [state.prefix + infix + suffix for state in automaton.states for suffix in char_set for infix in infixes]
