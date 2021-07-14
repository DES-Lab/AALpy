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

    def __init__(self, alphabet: list, sul: SUL, k: int = 2, method='random', target_coverage: float = 1,
                 num_generate_paths: int = 2000, max_path_len: int = 50, minimize_paths: bool = False,
                 optimize: str = 'steps', random_walk_len=10):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            k: k value used for K-Way transitions, i.e the number of steps between the start and the end of a transition
            target_coverage: percent of the minimum coverage that should be achieved  
            num_generate_paths: number of random queries used to find the optimal subset
            max_path_len: the maximum step size of a generated path
            minimize_paths: if true the generated paths will be trimmed in front and back if there is no coverage value
            optimize: minimize either the number of  'steps' or 'queries' that are executed
        """
        super().__init__(alphabet, sul)
        assert k >= 2
        assert method in ['random', 'prefix']
        assert 0 <= target_coverage <= 1
        assert optimize in ['steps', 'queries']

        self.k = k
        self.method = method
        self.target_coverage = target_coverage
        self.num_generate_paths = num_generate_paths
        self.max_path_len = max_path_len
        self.minimize_paths = minimize_paths
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

        size_of_universe = len(hypothesis.states) * pow(len(self.alphabet), self.k)
        size_of_target_coverage = int(size_of_universe * self.target_coverage)

        while size_of_target_coverage > len(covered):
            path = self.select_optimal_path(covered, paths)

            if path is not None:
                covered = set.union(covered, path.kWayTransitions)
                paths.remove(path)
                result.append(path)

            if self.minimize_paths:
                paths = self.get_minimized_paths(hypothesis, covered, paths)

            if path is None or not paths:
                print(f'Generate Paths: {size_of_target_coverage:10} ==> {len(covered):10d}')
                paths = [self.create_path(hypothesis, steps) for steps in self.generate_prefix_steps(hypothesis)]

        return result

    def select_optimal_path(self, covered: set, paths: list) -> Path:
        result = None

        if self.optimize == 'steps':
            result = max(paths, key=lambda p: len(
                p.kWayTransitions - covered) / len(p.steps))

        if self.optimize == 'queries':
            result = max(paths, key=lambda p: len(p.kWayTransitions - covered))

        return result if len(result.kWayTransitions - covered) != 0 else None

    def get_minimized_paths(self, hypothesis, covered, paths):
        result = list()

        for path in paths:
            if len(path.kWayTransitions - covered):
                path = self.trim_path_front(hypothesis, covered, path)
                path = self.trim_path_back(hypothesis, covered, path)
                result.append(path)

        return result

    def trim_path_front(self, hypothesis, covered: set, path: Path) -> Path:
        for i, transition in enumerate(path.transitions_log):
            if transition in covered:
                if i == 0:
                    return path

                prefix = hypothesis.get_state_by_id(transition.start_state).prefix
                steps = prefix + path.steps[i:]
                return self.create_path(hypothesis, steps)

        return path

    def trim_path_back(self, hypothesis, covered: set, path: Path) -> Path:
        for i, transition in enumerate(reversed(path.transitions_log)):
            if transition in covered:
                if i == 0:
                    return path

                prefix = hypothesis.get_state_by_id(transition.start_state).prefix
                steps = prefix + path.steps[:-i]
                return self.create_path(hypothesis, steps)

        return path

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
