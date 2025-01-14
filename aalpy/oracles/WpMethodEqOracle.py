from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from itertools import product, chain


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


class WpMethodEqOracle(Oracle):
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

        middle = []
        for i in range(min(self.m + 1 - len(hypothesis.states), 3)):
            middle.extend(list(product(self.alphabet, repeat=i)))
        first_phase = product(state_cover, middle, hypothesis.characterization_set)

        # the second_phase consists of the transition in the transition_cover
        # that are not in the state_cover, concatenated with the corresponding
        # characterization set of the state that the transition leads to.
        second_phase = set()
        for t, mid in product(difference, middle):
            _ = hypothesis.execute_sequence(hypothesis.initial_state, t + mid)
            state = hypothesis.current_state
            char_set = state_characterization_set(hypothesis, self.alphabet, state)
            concatenated = product([t], [mid], char_set)
            second_phase.update(concatenated)

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
