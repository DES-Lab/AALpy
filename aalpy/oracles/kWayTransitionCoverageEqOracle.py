from collections import namedtuple
from itertools import combinations, permutations, product
from random import choices, randint, shuffle

from aalpy.base import SUL, Automaton, Oracle

KWayTransition = namedtuple("KWayTransition", "start_state end_state steps")
Path = namedtuple("Path", "start_state end_state steps kWayTransitions, transitions_log")


class KWayTransitionCoverageEqOracle(Oracle):
    """
    This Equivalence oracle selects test cases based on K-Way transitions coverage. It does that
    by generating random queries and finding the smallest subset with the highest coverage. 
    """

    def __init__(self, alphabet: list, sul: SUL, k: int = 2, target_coverage: float = 0.8, num_generate_paths: int = 20000, refills: int = 5, max_path_len: int = 50, minimize_paths: bool = True, optimize: str = 'steps'):
        """
        Args:
            alphabet: input alphabet
            sul: system under learning
            k: k value used for K-Way transitions, i.e the number of steps between the start and the end of a transition
            target_coverage: percent of the minimum coverage that should be achieved  
            num_generate_paths: number of random queries used to find the optimal subset
            refills: number of refills that can happen if the target coverage is not reached
            max_path_len: the maximum step size of a generated path
            minimize_paths: if true the generated paths will be trimed in front and back if there is no coverage value
            optimize: minimize either the number of  'steps' or 'queries' that are executed
        """
        super().__init__(alphabet, sul)
        assert k >= 2
        assert 0 < target_coverage and target_coverage <= 1
        assert optimize in ['steps', 'queries']

        self.k = k
        self.target_coverage = target_coverage
        self.num_generate_paths = num_generate_paths
        self.refills = refills
        self.max_path_len = max_path_len
        self.minimize_paths = minimize_paths
        self.optimize = optimize
        self.cached_paths = list()

    def find_cex(self, hypothesis: Automaton):
        random_paths = self.generate_random_paths(hypothesis)

        selected_paths, refilled = self.greedy_set_cover(
            hypothesis, random_paths + self.cached_paths, self.refills)

        if refilled:
            selected_paths, _ = self.greedy_set_cover(
                hypothesis, selected_paths, 0)

        self.cached_paths = selected_paths

        for path in selected_paths:
            counter_example = self.check_path(hypothesis, path)

            if counter_example is not None:
                return counter_example

        return None

    def greedy_set_cover(self, hypothesis: Automaton, paths: list, max_refills: int) -> list:
        result = list()
        covered = set()

        refills = 0

        size_of_universe = len(hypothesis.states) * \
            pow(len(self.alphabet), self.k)
        size_of_target_coverage = int(size_of_universe * self.target_coverage)

        while size_of_target_coverage >= len(covered):
            path = self.select_optimal_path(covered, paths)

            if path is not None:
                covered = set.union(covered, path.kWayTransitions)
                paths.remove(path)
                result.append(path)

            if self.minimize_paths:
                paths = self.get_minimized_paths(hypothesis, covered, paths)

            if path is None or not paths:
                if refills >= max_refills:
                    break

                refills += 1
                paths = self.generate_random_paths(hypothesis)

        return result, refills == 0

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
            end_state = end_states[i+self.k - 1]
            chunk = tuple(steps[i:i+self.k])

            transition = KWayTransition(prev_state.state_id, end_state.state_id, chunk)

            transitions_log.append(transition)
            transitions.add(transition)

        return Path(hypothesis.initial_state, end_states[-1], steps, transitions, transitions_log)

    def check_path(self, hypothesis: Automaton, path: Path):
        self.reset_hyp_and_sul(hypothesis)

        for i, s in enumerate(path.steps):
            out_sul = self.sul.step(s)
            out_hyp = hypothesis.step(s)

            self.num_steps += 1

            if out_sul != out_hyp:
                return path.steps[:i + 1]

        return None
