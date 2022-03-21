from abc import ABC, abstractmethod
from random import randint, choice

from aalpy.learning_algs import run_Alergia


class Sampler(ABC):
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

            Data to be added to the data set for the passive learnign.

        """
        pass


class RandomWordSampler(Sampler):
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

            sample = [outputs.pop(0)]
            for i in range(len(outputs)):
                sample.append((random_walk[i], outputs[i]))

            samples.append(sample)

        return samples


def run_active_Alergia(data, sul, sampler, n_iter, eps=0.005, compatibility_checker=None, automaton_type='mdp',
                       print_info=True):
    """
    Active version of IOAlergia algorithm. Based on intermediate hypothesis sampling on the system is performed.
    Sampled data is added to the learning data and more accurate model is learned.
    Proposed in "Aichernig and Tappler, Probabilistic Black-Box Reachability Checking"

    Args:

        data: initial learning data, in form [[O, (I,O), (I,O)...] ,...] where O is outputs and I input.
        sul: system under learning which is basis for sampling
        sampler: instance of Sampler class
        n_iter: number of iterations of active learning
        eps: epsilon value if the default checker is used. Look in run_Alergia for description
        compatibility_checker: passed to run_Alergia, check there for description
        automaton_type: either 'mdp' or 'smm' (Markov decision process or Stochastic Mealy Machine)
        print_info: print current learning iteration

    Returns:

        learned MDP

    """
    model = None
    for i in range(n_iter):
        if print_info:
            print(f'Active Alergia Iteration: {i}')
        model = run_Alergia(data, automaton_type='mdp', eps=eps, compatibility_checker=compatibility_checker)

        new_samples = sampler.sample(sul, model)
        data.extend(new_samples)

    return model

