from collections import namedtuple
from itertools import product
from random import choices, randint, random

from aalpy.base import SUL, Automaton, Oracle

KWayTransition = namedtuple("KWayTransition", "start_state end_state steps")
Path = namedtuple("Path", "start_state end_state steps kWayTransitions, transitions_log")


class KWayTransitionCoverageEqOracle(Oracle):
    """
    This Equivalence oracle selects test cases based on k-way transitions coverage. It does that
    by generating random queries and finding the smallest subset with the highest coverage. In other words, this oracle
    finds counter examples by running random paths that cover all pairwise / k-way transitions.
    """

    def __init__(self, alphabet: list, sul: SUL, k: int = 2, method='random',
                 num_generate_paths: int = 20000,
                 max_path_len: int = 50,
                 max_number_of_steps: int = 0,
                 optimize: str = 'steps',
                 random_walk_len=10):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            k: k value used for K-Way transitions, i.e the number of steps between the start and the end of a transition
            method: defines how the queries are generated 'random' or 'prefix'
            num_generate_paths: number of random queries used to find the optimal subset
            max_path_len: the maximum step size of a generated path
            max_number_of_steps: maximum number of steps that will be executed on the SUL (0 = no limit)
            optimize: minimize either the number of  'steps' or 'queries' that are executed
            random_walk_len: the number of steps that are added by 'prefix' generated paths

        """
        super().__init__(alphabet, sul)
        assert k >= 2
        assert method in ['random', 'prefix']
        assert optimize in ['steps', 'queries']

        self.k = k
        self.method = method
        self.num_generate_paths = num_generate_paths
        self.max_path_len = max_path_len
        self.max_number_of_steps = max_number_of_steps
        self.optimize = optimize
        self.random_walk_len = random_walk_len

        self.cached_paths = list()

    def find_cex(self, hypothesis: Automaton):
        if self.method == 'random':
            paths = self.generate_random_paths(hypothesis) + self.cached_paths
            self.cached_paths = self.greedy_set_cover(hypothesis, paths)

            for path in self.cached_paths:
                counter_example = self.check_path(hypothesis, path.steps)

                if counter_example is not None:
                    return counter_example

        elif self.method == 'prefix':
            for steps in self.generate_prefix_steps(hypothesis):
                counter_example = self.check_path(hypothesis, steps)

                if counter_example is not None:
                    return counter_example
        return None

    def greedy_set_cover(self, hypothesis: Automaton, paths: list):
        result = list()
        covered = set()
        step_count = 0

        size_of_universe = len(hypothesis.states) * pow(len(self.alphabet), self.k)

        while size_of_universe > len(covered):
            path = self.select_optimal_path(covered, paths)

            if path is not None:
                covered = set.union(covered, path.kWayTransitions)
                paths.remove(path)
                result.append(path)
                step_count += len(path.steps)

            if path is None or not paths:
                paths = [self.create_path(hypothesis, steps) for steps in self.generate_prefix_steps(hypothesis)]

            if self.max_number_of_steps != 0 and step_count > self.max_number_of_steps:
                print("stop")
                break

        return result

    def select_optimal_path(self, covered: set, paths: list) -> Path:
        result = None

        if self.optimize == 'steps':
            result = max(paths, key=lambda p: len(
                p.kWayTransitions - covered) / len(p.steps))

        if self.optimize == 'queries':
            result = max(paths, key=lambda p: len(p.kWayTransitions - covered))

        return result if len(result.kWayTransitions - covered) != 0 else None

    def generate_random_paths(self, hypothesis: Automaton) -> list:
        result = list()

        for _ in range(self.num_generate_paths):
            random_length = randint(self.k, self.max_path_len)
            steps = tuple(choices(self.alphabet, k=random_length))
            path = self.create_path(hypothesis, steps)
            result.append(path)

        return result

    def generate_prefix_steps(self, hypothesis: Automaton) -> tuple:
        for state in reversed(hypothesis.states):
            prefix = state.prefix
            for steps in sorted(product(self.alphabet, repeat=self.k), key=lambda k: random()):
                yield prefix + steps + tuple(choices(self.alphabet, k=self.random_walk_len))

    def create_path(self, hypothesis: Automaton, steps: tuple) -> Path:
        transitions = set()
        transitions_log = list()

        prev_states = list()
        end_states = list()

        hypothesis.reset_to_initial()

        for i, s in enumerate(steps):
            prev_states.append(hypothesis.current_state)
            hypothesis.step(s)
            end_states.append(hypothesis.current_state)

        for i in range(len(steps) - self.k + 1):
            prev_state = prev_states[i]
            end_state = end_states[i + self.k - 1]
            chunk = tuple(steps[i:i + self.k])

            transition = KWayTransition(prev_state.state_id, end_state.state_id, chunk)

            transitions_log.append(transition)
            transitions.add(transition)

        return Path(hypothesis.initial_state, end_states[-1], steps, transitions, transitions_log)

    def check_path(self, hypothesis: Automaton, steps: tuple):
        self.reset_hyp_and_sul(hypothesis)

        for i, s in enumerate(steps):
            out_sul = self.sul.step(s)
            out_hyp = hypothesis.step(s)

            self.num_steps += 1

            if out_sul != out_hyp:
                self.sul.post()
                return steps[:i + 1]

        return None
