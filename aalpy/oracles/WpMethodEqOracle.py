from itertools import chain, tee

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from aalpy.utils.HelperFunctions import product_with_possible_empty_iterable


def state_characterization_set(hypothesis, alphabet, state):
    """
    Return a list of sequences that distinguish the given state from all other states in the hypothesis.
    Args:
        hypothesis: hypothesis automaton
        alphabet: input alphabet
        state: state for which to find distinguishing sequences
    """
    result = []
    for i in range(len(hypothesis.states)):
        if hypothesis.states[i] == state:
            continue
        seq = hypothesis.find_distinguishing_seq(state, hypothesis.states[i], alphabet)
        if seq:
            result.append(tuple(seq))
    return result


def i_star(alphabet, max_seq_len):
    """
    Return an iterator that generates all possible sequences of length upto from the given alphabet.
    Args:
        alphabet: input alphabet
        max_seq_len: maximum length of the sequences
    """
    return chain(*(product_with_possible_empty_iterable(alphabet, repeat=i) for i in range(max_seq_len)))

def first_phase_it(alphabet, state_cover, depth, char_set):
    """
    Return an iterator that generates all possible sequences for the first phase of the Wp-method.
    Args:
        alphabet: input alphabet
        state_cover: list of states to cover
        depth: maximum length of middle part
        char_set: characterization set
    """
    for d in range(depth):
        middle = product_with_possible_empty_iterable(alphabet, repeat=d)
        for m in middle:
            for case in product_with_possible_empty_iterable(state_cover, [m], char_set):
                yield case

def second_phase_it(hyp, alphabet, difference, depth):
    """
    Return an iterator that generates all possible sequences for the second phase of the Wp-method.
    Args:
        hyp: hypothesis automaton
        alphabet: input alphabet
        difference: set of sequences that are in the transition cover but not in the state cover
        depth: maximum length of middle part
    """
    state_mapping = {}
    for d in range(depth):
        middle = product_with_possible_empty_iterable(alphabet, repeat=d)
        for t, mid in product_with_possible_empty_iterable(difference, middle):
            _ = hyp.execute_sequence(hyp.initial_state, t + mid)
            state = hyp.current_state
            if state not in state_mapping:
                state_mapping[state] = state_characterization_set(hyp, alphabet, state)

            for sm in state_mapping[state]:
                yield (t,) + (mid,) + (sm,)


class WpMethodEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states=4):
        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.cache = set()

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        # keep this as a list, because we only iterate over it
        transition_cover = [
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        ]

        state_cover = [state.prefix for state in hypothesis.states]
        # make the state cover a set for faster lookup
        difference = [tc for tc in transition_cover if tc not in set(state_cover)]
        depth = self.m + 1 - len(hypothesis.states)
        # first phase State Cover * Middle * Characterization Set
        first_phase = first_phase_it(self.alphabet, state_cover, depth, hypothesis.characterization_set)

        # second phase (Transition Cover - State Cover) * Middle * Characterization Set
        # of the state that the prefix leads to
        second_phase = second_phase_it(hypothesis, self.alphabet, difference, depth)
        test_suite = chain(first_phase, second_phase)

        for seq in test_suite:
            seq = tuple([i for sub in seq for i in sub])

            if seq not in self.cache:
                self.reset_hyp_and_sul(hypothesis)

                for ind, letter in enumerate(seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1

                    if out_hyp != out_sul:
                        self.sul.post()
                        return seq[: ind + 1]
                self.cache.add(seq)

        return None
