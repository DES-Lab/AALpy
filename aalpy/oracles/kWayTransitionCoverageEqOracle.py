from random import choices, shuffle, randint

from collections import namedtuple

from itertools import combinations, permutations, product

from aalpy.base import Automaton, Oracle, SUL

import math

KWayTransition = namedtuple(
    "KWayTransition", "start_state end_state steps")
Path = namedtuple("Path", "start_state end_state steps kWayTransitions")


class KWayTransitionCoverageEqOracle(Oracle):
    """
    WIP
    """

    def __init__(self, alphabet: list, sul: SUL, k: int = 2, target_cover: float = 1.0, refills: int = 10):
        super().__init__(alphabet, sul)
        assert k >= 2
        assert 0 < target_cover and target_cover < 1

        self.k = k
        self.target_cover = target_cover
        self.refills = refills

    def find_cex(self, hypothesis: Automaton):
        random_paths = self.generate_random_paths(hypothesis)

        selected_paths, refilled = self.greedy_set_cover(
            hypothesis, random_paths, self.refills)

        selected_paths, _ = self.greedy_set_cover(
            hypothesis, selected_paths, 0) if refilled else selected_paths

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
        size_of_target_cover = int(size_of_universe * self.target_cover)

        print("Size of universe: " + str(size_of_universe))
        print("Size of targed cover: " + str(size_of_target_cover))

        while size_of_target_cover >= len(covered):
            path = self.select_optimal_path(covered, paths)

            if path is not None:
                covered = set.union(covered, path.kWayTransitions)
                paths.remove(path)
                result.append(path)

            if path is None or not paths:
                print("Len of covered: " + str(len(covered)))
                print("Len of missing: " +
                      str(size_of_target_cover - len(covered)))

                paths = self.generate_random_paths(hypothesis)

                refills += 1
                if refills >= max_refills:
                    break

        print("Len of greedy result: " + str(len(result)))
        return result, refills == 0

    def generate_random_paths(self, hypothesis: Automaton, number_of_paths: int = 8000, min_len: int = 2, max_len: int = 50) -> list:
        result = list()

        for _ in range(number_of_paths):
            steps = tuple(
                choices(self.alphabet, k=randint(min_len, max_len)))
            path = self.create_path(hypothesis, steps)
            result.append(path)

        print("Number of paths generated: " + str(len(result)))
        return result

    def create_path(self, hypothesis: Automaton, steps: tuple) -> Path:
        transitions = set()

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

            transitions.add(KWayTransition(
                prev_state.state_id, end_state.state_id, chunk))

        return Path(hypothesis.initial_state, end_states[-1], steps, transitions)

    def select_optimal_path(self, covered: set, paths: list) -> Path:
        # Note: Optimize number of steps
        result = max(paths, key=lambda p: len(
            p.kWayTransitions-covered)/len(p.steps))

        return result if len(result.kWayTransitions - covered) != 0 else None

    def check_path(self, hypothesis: Automaton, path: Path):
        self.reset_hyp_and_sul(hypothesis)

        for i, s in enumerate(path.steps):
            out_sul = self.sul.step(s)
            out_hyp = hypothesis.step(s)

            self.num_steps += 1

            if out_sul != out_hyp:
                return path.steps[:i + 1]

        return None

    # ---- Not used anymore

    def create_universe(self, hypothesis: Automaton) -> set:
        '''
        Returns a list of all possible KWayTranstions from an Automaton
        '''
        result = set()

        print("Len of states: " + str(len(hypothesis.states)))
        print("Len of alphabet: " + str(len(self.alphabet)))

        for start_state in hypothesis.states:
            print(len(product(self.alphabet, repeat=self.k)))
            for steps in product(self.alphabet, repeat=self.k):
                hypothesis.reset_to_initial()
                for s in start_state.prefix + steps:
                    hypothesis.step(s)

                newKWayTranstions = KWayTransition(
                    start_state.state_id, hypothesis.current_state.state_id, steps)
                result.add(newKWayTranstions)

        print("Len of universe: " + str(len(result)))
        return result

    def create_universe_paths(self, hypothesis, universe):
        result = list()

        for transition in universe:
            for state in hypothesis.states:
                if state.state_id == transition.start_state:
                    path = self.create_path(
                        hypothesis, state.prefix + transition.steps)
                    result.append(path)

        return result
