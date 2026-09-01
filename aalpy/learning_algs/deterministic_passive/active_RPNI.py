# Active sampling wrapper around RPNI that iteratively augments the passive learning data set with new samples.
from abc import ABC, abstractmethod
from random import randint, choice

from aalpy.base import SUL, DeterministicAutomaton
from aalpy.learning_algs import run_RPNI
from aalpy.utils import convert_i_o_traces_for_RPNI


class RpniActiveSampler(ABC):
    """
    Abstract class whose implementations are used to provide samples for active passive learning.
    """

    @abstractmethod
    def sample(self, sul: SUL, model: DeterministicAutomaton) -> list:
        """
        Abstract method implementing sampling strategy.

        :param SUL sul: system under learning
        :param DeterministicAutomaton model: current learned model
        :return list: Data to be added to the data set for the passive RPNI learning in its data-format.
        """
        pass


class RandomWordSampler(RpniActiveSampler):
    """
    Sampling strategy that queries the SUL with randomly generated words of random length.
    """

    def __init__(self, num_walks: int, min_walk_len: int, max_walk_len: int) -> None:
        """
        Creates a random word sampler.

        :param int num_walks: Number of random walks to perform per sampling call.
        :param int min_walk_len: Minimum length of a random walk.
        :param int max_walk_len: Maximum length of a random walk.
        """
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len

    def sample(self, sul: SUL, model: DeterministicAutomaton) -> list:
        """
        Samples the SUL with random walks over the input alphabet inferred from the current model.

        :param SUL sul: System under learning to query.
        :param DeterministicAutomaton model: Current learned model, used to determine the input alphabet.
        :return list: Data to be added to the data set for the passive RPNI learning in its data-format.
        """
        input_al = list({el for s in model.states for el in s.transitions.keys()})
        samples = []

        for _ in range(self.num_walks):
            walk_len = randint(self.min_walk_len, self.max_walk_len)
            random_walk = tuple(choice(input_al) for _ in range(walk_len))

            outputs = sul.query(random_walk)
            samples.append(list(zip(random_walk, outputs)))

        samples = convert_i_o_traces_for_RPNI(samples)
        return samples


def run_active_RPNI(data: list, sul: SUL, sampler: RpniActiveSampler, n_iter: int, automaton_type: str,
                     print_info: bool = True) -> DeterministicAutomaton | None:
    """
    Runs RPNI iteratively, extending the data set after each iteration with new samples obtained from the SUL.

    :param list data: Initial sequence of input sequences and corresponding labels.
    :param SUL sul: System under learning queried by the sampler to obtain new samples.
    :param RpniActiveSampler sampler: Sampling strategy used to generate new data between iterations.
    :param int n_iter: Number of iterations to perform.
    :param str automaton_type: Either 'dfa', 'mealy', or 'moore'.
    :param bool print_info: Whether to print learning progress and runtime information.
    :return DeterministicAutomaton | None: The model learned in the final iteration, or None if the data is
        non-deterministic.
    """
    model = None
    for i in range(n_iter):
        if print_info:
            print(f'-------------Active RPNI Iteration: {i}-------------')
        model = run_RPNI(data, automaton_type=automaton_type, print_info=print_info)

        new_samples = sampler.sample(sul, model)
        data.extend(new_samples)

    return model
