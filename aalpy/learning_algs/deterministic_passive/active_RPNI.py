from abc import ABC, abstractmethod
from random import randint, choice

from aalpy.learning_algs import run_RPNI
from aalpy.utils import convert_i_o_traces_for_RPNI


class RpniActiveSampler(ABC):
    """
    Abstract class whose implementations are used to provide samples for active passive learning.
    """

    @abstractmethod
    def sample(self, sul, model):
        """
        Abstract method implementing sampling strategy.

        Args:

            sul: system under learning
            model: current learned model

        Returns:

            Data to be added to the data set for the passive RPNI learning in its data-format.

        """
        pass


class RandomWordSampler(RpniActiveSampler):
    def __init__(self, num_walks, min_walk_len, max_walk_len):
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len

    def sample(self, sul, model):
        input_al = list({el for s in model.states for el in s.transitions.keys()})
        samples = []

        for _ in range(self.num_walks):
            walk_len = randint(self.min_walk_len, self.max_walk_len)
            random_walk = tuple(choice(input_al) for _ in range(walk_len))

            outputs = sul.query(random_walk)
            samples.append(list(zip(random_walk, outputs)))

        samples = convert_i_o_traces_for_RPNI(samples)
        return samples


def run_active_RPNI(data, sul, sampler, n_iter, automaton_type, print_info=True):
    model = None
    for i in range(n_iter):
        if print_info:
            print(f'-------------Active RPNI Iteration: {i}-------------')
        model = run_RPNI(data, automaton_type=automaton_type, print_info=print_info)

        new_samples = sampler.sample(sul, model)
        data.extend(new_samples)

    return model
