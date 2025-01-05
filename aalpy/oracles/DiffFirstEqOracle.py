import random
from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from itertools import dropwhile


def get_loop_degrees(model):
    al = model.get_input_alphabet()
    states = model.states
    loops = [s.get_same_state_transitions() for s in states]
    degrees = [round(len(l) / len(al), 3) for l in loops]
    return degrees


def count_drops(array):
    index = 0
    while array[index] == 0:
        index += 1
    normal = array[index:]
    counter = 0
    index = 0
    while index <= len(normal) - 1:
        pivot = normal[index]
        while index <= len(normal) - 1 and pivot <= normal[index]:
            index += 1
        if index > len(normal) - 1:
            break
        else:
            counter += 1
    return counter


class DiffFirstEqOracle(Oracle):
    """
    Equivalence oracle that first explores the 'difference' of the current hypothesis to the previous
    one. The intuition behind this is that, if a new error is to occur, then it is likely to occur in the
    'new' part of the hypothesis, that has not been examined before.
    """

    def __init__(self, alphabet, sul: SUL, walks_per_round, walk_len):

        super().__init__(alphabet, sul)
        self.walks_per_round = walks_per_round
        self.walk_len = walk_len
        self.age_groups = []

    def find_cex(self, hypothesis):
        # identify new states
        new_states = []
        for state in hypothesis.states:
            if state not in hypothesis.states:
                new_states.append(state)

        # sort them by the number of same state transitions
        new_states.sort(key=lambda x: len(x.get_same_state_transitions()))
        self.age_groups.append(new_states)

        # new_states is a set of possibly more than 1 state. These states
        # possibly have loops and 2 cases can be distinguished:
        #
        # 1) Each state in the group has a small amount of loops and the group
        # contains possibly 1 sink state
        #
        # 2) The states have a medium amount of loops with no sink state
        #
        # In the first case, the sink state is some sort of unrecoverable
        # error. In the second case, the groups make up a chain and which
        # possibly represents progress in the protocol.
        #
        # If we determine that the group resembles a chain, we must focus on
        # transitions, whereas in the other case we should attack the sink
        # state.
        #
