from random import choice


class hWOracle:
    """
    Base class for hW resetless equivalence oracles.

    An oracle drives the (resetless) SUL through the learner's step_wrapper and
    looks for a divergence between the SUL and the current hypothesis. As there are
    no resets, every executed input is part of one continuous run and is recorded in
    the learner's global trace.

    Oracles are constructed with their configuration only and handed to run_hW; the
    hW learner binds itself to the oracle (sets self.learner) before learning starts,
    so a single oracle instance must not be shared between concurrent runs.
    """

    def __init__(self):
        self.learner = None
        self.num_steps = 0  # total SUL steps executed by this oracle across all checks

    def find_counterexample(self, hypothesis):
        """
        Return the executed inputs up to and including the first output mismatch
        with the hypothesis, or None if no mismatch was observed.
        """
        raise NotImplementedError

    def _execute_and_compare(self, hypothesis, inputs, cex):
        """
        Run inputs on the SUL and the hypothesis in lock-step, appending each to
        cex. Returns True on the first output mismatch (cex then ends at the
        diverging input).
        """
        learner = self.learner
        for i in inputs:
            self.num_steps += 1
            cex.append(i)
            if learner.step_wrapper(i) != hypothesis.step(i):
                return True
        return False


class RandomhWOracle(hWOracle):
    """
    Random-walk equivalence check: take random inputs until the SUL and the
    hypothesis disagree or the per-round step budget is exhausted.

    Args:
        num_testing_steps: number of random steps used per equivalence check
        reset_testing_counter: if True, the step budget is reset for every
            equivalence check; otherwise num_testing_steps bounds the total number
            of testing steps across the whole run
    """

    def __init__(self, num_testing_steps=200, reset_testing_counter=True):
        super().__init__()
        self.num_testing_steps = num_testing_steps
        self.reset_testing_counter = reset_testing_counter

    def find_counterexample(self, hypothesis):
        learner = self.learner
        if self.reset_testing_counter:
            current_test_steps = self.num_testing_steps
        else:
            current_test_steps = max(self.num_testing_steps - self.num_steps, 0)

        cex = []
        for _ in range(current_test_steps):
            random_input = choice(learner.input_alphabet)
            if self._execute_and_compare(hypothesis, (random_input,), cex):
                return cex

        return None


class RandomWphWOracle(hWOracle):
    """
    Resetless variant of random Wp-method testing. For num_test_origin_states
    iterations a random hypothesis state is selected as origin, reached from the
    hypothesis' current state, and then probed with a random element of the
    characterization set W followed by a random walk of length random_walk_length.
    The first such probe that diverges from the hypothesis is returned.
    """

    def __init__(self, random_walk_length=20, num_test_origin_states=10):
        super().__init__()
        self.random_walk_length = random_walk_length
        self.num_test_origin_states = num_test_origin_states

    def find_counterexample(self, hypothesis):
        learner = self.learner

        for _ in range(self.num_test_origin_states):
            cex = []

            # reach a random origin state from the hypothesis' current state
            target = choice(hypothesis.states)
            path = hypothesis.get_shortest_path(hypothesis.current_state, target)
            if path is None:
                path = ()
            if self._execute_and_compare(hypothesis, path, cex):
                return cex

            # distinguish the reached state with a random element of W
            if learner.W:
                w = choice(learner.W)
                if self._execute_and_compare(hypothesis, w, cex):
                    return cex

            # explore further with a random walk from the reached state
            walk = [choice(learner.input_alphabet) for _ in range(self.random_walk_length)]
            if self._execute_and_compare(hypothesis, walk, cex):
                return cex

        return None


def _trace_step_explained_by(learner, state, letter, output):
    """Target state if the hypothesis state reproduces this trace step, else None."""
    target = state.transitions.get(letter)
    if target is None:
        return None
    predicted = target.output if learner.is_moore else state.output_fun.get(letter)
    return target if predicted == output else None


def find_counterexample_in_trace(learner, hypothesis):
    """
    Search the already-observed global trace for a sub-trace that no hypothesis
    state can explain (Sec. 6.1 of the hW paper); used as a free backstop when
    the equivalence oracle finds no counterexample. Returns the inputs of such a
    sub-trace, or None.

    A single forward pass suffices: if some state explains trace[0..j], its
    intermediate states explain every sub-trace of it, so the alive set only
    empties if an unexplained sub-trace exists.
    """
    alive = set(hypothesis.states)
    for j, (letter, output) in enumerate(learner.global_trace):
        alive = {t for q in alive
                 if (t := _trace_step_explained_by(learner, q, letter, output)) is not None}
        if not alive:
            return _shorten_trace_counterexample(learner, hypothesis.states, j)
    return None


def _shorten_trace_counterexample(learner, states, end):
    """Inputs of trace [i..end] for the latest start i that no state can explain."""
    trace = learner.global_trace
    explaining = set(states)  # states explaining trace[i+1..end]
    start = 0
    for i in range(end, -1, -1):
        letter, output = trace[i]
        explaining = {q for q in states
                      if _trace_step_explained_by(learner, q, letter, output) in explaining}
        if not explaining:
            start = i
            break
    return [letter for letter, _ in trace[start:end + 1]]
