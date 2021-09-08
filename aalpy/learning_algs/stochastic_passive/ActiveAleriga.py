from abc import ABC, abstractmethod
from random import randint, choice

from aalpy.learning_algs import run_Alergia


class Sampler(ABC):

    @abstractmethod
    def sample(self, sul, model):
        pass


class RandomWordSampler(Sampler):
    def __init__(self, num_walks, min_walk_len, max_walk_len):
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len

    def sample(self, sul, model):
        samples = []
        for _ in range(self.num_walks):
            walk_len = randint(self.min_walk_len, self.max_walk_len)
            random_walk = tuple(choice(model.get_input_alphabet()) for _ in range(walk_len))

            outputs = sul.query(random_walk)

            sample = [outputs.pop(0)]
            for i in range(len(outputs)):
                sample.append((random_walk[i], outputs[i]))

            samples.append(sample)

        return samples


def run_active_Alergia(data, sul, sampler, n_iter):
    model = None
    for _ in range(n_iter):
        model = run_Alergia(data, automaton_type='mdp')

        new_samples = sampler.sample(sul, model)
        data.extend(new_samples)

    return model

