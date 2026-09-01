# Active Alergia: samples from the system under learning based on intermediate hypotheses to augment the
# learning data used by (passive) Alergia/IOAlergia.
from abc import ABC, abstractmethod
from random import randint, choice

from aalpy.automata import Mdp
from aalpy.base import SUL
from aalpy.learning_algs import run_Alergia
from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import CompatibilityChecker


class Sampler(ABC):
    """
    Abstract class whose implementations are used to provide samples for active passive learning.
    """

    @abstractmethod
    def sample(self, sul: SUL, model: Mdp) -> list:
        """
        Abstract method implementing sampling strategy.

        :param SUL sul: System under learning.
        :param Mdp model: Current learned model.
        :return list: Data to be added to the data set for the passive learning.
        """
        pass


class RandomWordSampler(Sampler):
    """
    Sampler that generates random walks over the input alphabet of the current hypothesis.
    """

    def __init__(self, num_walks: int, min_walk_len: int, max_walk_len: int) -> None:
        """
        Creates a random word sampler.

        :param int num_walks: Number of random walks to sample per iteration.
        :param int min_walk_len: Minimum length of a random walk.
        :param int max_walk_len: Maximum length of a random walk.
        """
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len

    def sample(self, sul: SUL, model: Mdp) -> list:
        """
        Samples num_walks random walks of random length over the current hypothesis' input alphabet.

        :param SUL sul: System under learning.
        :param Mdp model: Current learned model, used to extract the input alphabet.
        :return list: List of sampled traces in the form [output, (input, output), ...].
        """
        input_al = list({el for s in model.states for el in s.transitions.keys()})
        samples = []

        for _ in range(self.num_walks):
            walk_len = randint(self.min_walk_len, self.max_walk_len)
            random_walk = tuple(choice(input_al) for _ in range(walk_len))

            # the SUL's initial output (before any input) must be queried separately: sul.query(random_walk)
            # returns exactly one output per input in random_walk, none of which is the initial output
            initial_output = sul.query(())[0]
            outputs = sul.query(random_walk)

            sample = [initial_output]
            for i in range(len(outputs)):
                sample.append((random_walk[i], outputs[i]))

            samples.append(sample)

        return samples


def run_active_Alergia(data: list, sul: SUL, sampler: Sampler, n_iter: int, eps: float | str = 0.05,
                        compatibility_checker: CompatibilityChecker | None = None, automaton_type: str = 'mdp',
                        print_info: bool = True) -> Mdp:
    """
    Active version of IOAlergia algorithm. Based on intermediate hypothesis sampling on the system is performed.
    Sampled data is added to the learning data and more accurate model is learned.
    Proposed in "Aichernig and Tappler, Probabilistic Black-Box Reachability Checking".

    :param list data: Initial learning data, in form [[O, (I,O), (I,O)...] ,...] where O is outputs and I input.
    :param SUL sul: System under learning which is basis for sampling.
    :param Sampler sampler: Instance of Sampler class.
    :param int n_iter: Number of iterations of active learning.
    :param float | str eps: Epsilon value if the default checker is used. Look in run_Alergia for description.
    :param CompatibilityChecker | None compatibility_checker: Passed to run_Alergia, check there for description.
    :param str automaton_type: Either 'mdp' or 'smm' (Markov decision process or Stochastic Mealy Machine).
    :param bool print_info: Print current learning iteration.
    :return Mdp: Learned MDP.
    """
    model = None
    for i in range(n_iter):
        if print_info:
            print(f'Active Alergia Iteration: {i}')
        model = run_Alergia(data, automaton_type=automaton_type, eps=eps, compatibility_checker=compatibility_checker)

        new_samples = sampler.sample(sul, model)
        data.extend(new_samples)

    return model
