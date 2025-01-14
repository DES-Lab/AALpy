from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from itertools import product, chain, tee


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


def i_star(alph, upto):
    """
    Return an iterator that generates all possible sequences of length upto from the given alphabet.
    Args:
        alph: input alphabet
        upto: maximum length of the sequences
    """
    return chain(*(product(alph, repeat=i) for i in range(upto)))


def second_phase_it(hyp, alph, difference, middle):
    """
    Return an iterator that generates all possible sequences for the second phase of the Wp-method.
    Args:
        hyp: hypothesis automaton
        alph: input alphabet
        difference: set of sequences that are in the transition cover but not in the state cover
        middle: iterator that generates all possible sequences of length upto from the given alphabet
    """
    for t, mid in product(difference, middle):
        _ = hyp.execute_sequence(hyp.initial_state, t + mid)
        state = hyp.current_state
        char_set = state_characterization_set(hyp, alph, state)
        concatenated = product([t], [mid], char_set)
        for el in concatenated:
            yield el


class WpMethodEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states):
        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.cache = set()

    def find_cex(self, hypothesis):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        transition_cover = set(
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        )
        state_cover = set(state.prefix for state in hypothesis.states)
        difference = transition_cover.difference(state_cover)

        # not really helpful but it's here
        minimum = min(self.m + 1 - len(hypothesis.states), 3)
        # two views of the same iterator
        middle_1, middle_2 = tee(i_star(self.alphabet, minimum), 2)
        # first phase State Cover * Middle * Characterization Set
        first_phase = product(state_cover, middle_1, hypothesis.characterization_set)
        # second phase (Transition Cover - State Cover) * Middle * Characterization Set
        # of the state that the prefix leads to
        second_phase = second_phase_it(hypothesis, self.alphabet, difference, middle_2)
        test_suite = chain(first_phase, second_phase)
        for seq in test_suite:
            inp_seq = tuple([i for sub in seq for i in sub])
            if inp_seq not in self.cache:
                self.reset_hyp_and_sul(hypothesis)
                outputs = []

                for ind, letter in enumerate(inp_seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1

                    outputs.append(out_sul)
                    if out_hyp != out_sul:
                        self.sul.post()
                        return inp_seq[: ind + 1]
                self.cache.add(inp_seq)

        return None
