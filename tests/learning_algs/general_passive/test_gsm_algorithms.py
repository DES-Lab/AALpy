import random
import unittest
from itertools import product

from aalpy.automata import (
    Dfa, DfaState, MooreMachine, MooreState, MealyMachine, MealyState, Mdp, MdpState, StochasticMealyMachine,
    StochasticMealyState,
)
from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs.general_passive.GsmAlgorithms import run_EDSM, run_Alergia_EDSM, run_k_tails
from aalpy.utils.ModelChecking import bisimilar


def even_a_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


def alternating_moore():
    q0 = MooreState('q0', 0)
    q1 = MooreState('q1', 1)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return MooreMachine(q0, [q0, q1])


def parity_mealy():
    q0 = MealyState('q0')
    q1 = MealyState('q1')
    q0.transitions = {'a': q1, 'b': q0}
    q0.output_fun = {'a': 'x', 'b': 'y'}
    q1.transitions = {'a': q0, 'b': q1}
    q1.output_fun = {'a': 'y', 'b': 'x'}
    return MealyMachine(q0, [q0, q1])


def labeled_sequence_data(automaton, depth=3):
    data = []
    alphabet = automaton.get_input_alphabet()
    is_mealy = isinstance(automaton, MealyMachine)
    start_level = 1 if is_mealy else 0
    for level in range(start_level, depth + 1):
        for seq in product(alphabet, repeat=level):
            automaton.reset_to_initial()
            if len(seq) == 0:
                label = automaton.initial_state.output
            else:
                outputs = automaton.execute_sequence(automaton.initial_state, seq)
                label = outputs[-1]
            data.append((seq, label))
    return data


class TestRunEdsm(unittest.TestCase):
    def test_learns_minimal_dfa(self):
        ground_truth = even_a_dfa()
        data = labeled_sequence_data(ground_truth, depth=3)
        learned = run_EDSM(data, automaton_type='dfa', print_info=False)
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))

    def test_learns_minimal_moore_machine(self):
        ground_truth = alternating_moore()
        data = labeled_sequence_data(ground_truth, depth=3)
        learned = run_EDSM(data, automaton_type='moore', print_info=False)
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))

    def test_learns_minimal_mealy_machine(self):
        ground_truth = parity_mealy()
        data = labeled_sequence_data(ground_truth, depth=3)
        learned = run_EDSM(data, automaton_type='mealy', print_info=False)
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))

    def test_input_completeness_sink_state(self):
        data = [((), True), (('a',), False), (('b',), True)]
        learned = run_EDSM(data, automaton_type='dfa', input_completeness='sink_state', print_info=False)
        self.assertTrue(learned.is_input_complete())

    def test_input_completeness_self_loop(self):
        data = [((), True), (('a',), False), (('b',), True)]
        learned = run_EDSM(data, automaton_type='dfa', input_completeness='self_loop', print_info=False)
        self.assertTrue(learned.is_input_complete())


def nd_moore_output_sequence_is_possible(machine, inputs, expected_outputs):
    """
    Checks whether there exists at least one path through the (possibly nondeterministic) Moore machine
    that reproduces the expected output sequence for the given inputs.
    """
    current_states = {machine.initial_state}
    for in_sym, expected_out in zip(inputs, expected_outputs):
        next_states = set()
        for state in current_states:
            for target in state.transitions.get(in_sym, []):
                if target.output == expected_out:
                    next_states.add(target)
        if not next_states:
            return False
        current_states = next_states
    return True


class TestRunKTails(unittest.TestCase):
    def io_traces_for_moore(self, ground_truth, depth=4):
        alphabet = ground_truth.get_input_alphabet()
        traces = []
        for level in range(1, depth + 1):
            for seq in product(alphabet, repeat=level):
                ground_truth.reset_to_initial()
                outputs = ground_truth.execute_sequence(ground_truth.initial_state, seq)
                traces.append([ground_truth.initial_state.output] + list(zip(seq, outputs)))
        return traces

    def test_learned_model_reproduces_training_traces_with_large_k(self):
        ground_truth = alternating_moore()
        traces = self.io_traces_for_moore(ground_truth, depth=4)

        learned = run_k_tails(traces, automaton_type='moore', k=10, print_info=False)

        for trace in traces:
            initial_output = trace[0]
            inputs = [i for i, _ in trace[1:]]
            outputs = [o for _, o in trace[1:]]
            self.assertEqual(learned.initial_state.output, initial_output)
            self.assertTrue(nd_moore_output_sequence_is_possible(learned, inputs, outputs),
                            f'trace {trace} not reproducible by learned k-tails model')

    def test_small_k_merges_more_aggressively_than_large_k(self):
        # with a small k, compatibility is only checked shallowly, so more (possibly behaviorally
        # different) states get merged, generally yielding a smaller (or equal) automaton than a
        # large k that checks compatibility much more thoroughly.
        ground_truth = alternating_moore()
        traces = self.io_traces_for_moore(ground_truth, depth=4)

        learned_small_k = run_k_tails(traces, automaton_type='moore', k=0, print_info=False)
        learned_large_k = run_k_tails(traces, automaton_type='moore', k=10, print_info=False)

        self.assertLessEqual(len(learned_small_k.states), len(learned_large_k.states))


class TestRunAlergiaEdsm(unittest.TestCase):
    def deterministic_mdp(self):
        """3-state MDP with only probability-1 transitions, so its behavior is effectively deterministic."""
        q0 = MdpState('q0', output='label')
        q1 = MdpState('q1', output='label')
        q2 = MdpState('q2', output='label')
        q0.transitions['a'] = [(q1, 1.0)]
        q0.transitions['b'] = [(q0, 1.0)]
        q1.transitions['a'] = [(q2, 1.0)]
        q1.transitions['b'] = [(q0, 1.0)]
        q2.transitions['a'] = [(q2, 1.0)]
        q2.transitions['b'] = [(q1, 1.0)]
        return Mdp(q0, [q0, q1, q2])

    def generate_traces(self, mdp, num_traces=200, max_len=6, seed=1):
        random.seed(seed)
        sul = AutomatonSUL(mdp)
        alphabet = mdp.get_input_alphabet()
        traces = []
        for _ in range(num_traces):
            length = random.randint(1, max_len)
            inputs = [random.choice(alphabet) for _ in range(length)]
            outputs = sul.query(tuple(inputs))
            trace = [mdp.initial_state.output]
            for i, o in zip(inputs, outputs):
                trace.append((i, o))
            traces.append(trace)
        return traces

    def test_learns_model_matching_deterministic_ground_truth_behavior(self):
        ground_truth = self.deterministic_mdp()
        traces = self.generate_traces(ground_truth)

        learned = run_Alergia_EDSM(traces, automaton_type='mdp', eps=0.05, print_info=False)

        random.seed(7)
        alphabet = ground_truth.get_input_alphabet()
        for _ in range(50):
            inputs = tuple(random.choice(alphabet) for _ in range(random.randint(1, 6)))

            ground_truth.reset_to_initial()
            expected = ground_truth.execute_sequence(ground_truth.initial_state, inputs)

            learned.reset_to_initial()
            actual = learned.execute_sequence(learned.initial_state, inputs)

            self.assertEqual(actual, expected, f'mismatch on {inputs}')


if __name__ == '__main__':
    unittest.main()
