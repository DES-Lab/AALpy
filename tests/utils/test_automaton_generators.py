import random
import unittest

from aalpy.automata import Dfa, MealyMachine, MooreMachine, Mdp, Onfsm, MarkovChain, Sevpa, StochasticMealyMachine
from aalpy.utils import (
    generate_random_deterministic_automata,
    generate_random_dfa,
    generate_random_mealy_machine,
    generate_random_moore_machine,
    generate_random_mdp,
    generate_random_smm,
    generate_random_ONFSM,
    generate_random_markov_chain,
    generate_random_sevpa,
)


def all_states_reachable(automaton):
    reached = set()
    to_visit = [automaton.initial_state]
    while to_visit:
        state = to_visit.pop()
        if state.state_id in reached:
            continue
        reached.add(state.state_id)
        for target in _successors(state):
            to_visit.append(target)
    return reached == {s.state_id for s in automaton.states}


def _successors(state):
    transitions = getattr(state, 'transitions', None)
    if transitions is None:
        return []
    if isinstance(transitions, list):
        return [target for target, _ in transitions]
    successors = []
    for value in transitions.values():
        if value is None:
            continue
        if isinstance(value, list):
            for entry in value:
                if hasattr(entry, 'target_state'):
                    successors.append(entry.target_state)
                elif isinstance(entry, tuple):
                    successors.append(entry[0] if not isinstance(entry[0], str) else entry[1])
        else:
            successors.append(value)
    return successors


class TestGenerateRandomDeterministicAutomata(unittest.TestCase):
    def test_dfa_structure(self):
        random.seed(1)
        dfa = generate_random_deterministic_automata('dfa', num_states=5, input_alphabet_size=3)
        self.assertIsInstance(dfa, Dfa)
        self.assertEqual(len(dfa.states), 5)
        self.assertEqual(len(dfa.get_input_alphabet()), 3)
        self.assertTrue(all_states_reachable(dfa))
        for state in dfa.states:
            self.assertEqual(set(state.transitions.keys()), set(dfa.get_input_alphabet()))

    def test_mealy_structure(self):
        random.seed(2)
        mealy = generate_random_deterministic_automata('mealy', num_states=4, input_alphabet_size=2,
                                                        output_alphabet_size=3)
        self.assertIsInstance(mealy, MealyMachine)
        self.assertEqual(len(mealy.states), 4)
        for state in mealy.states:
            for i in mealy.get_input_alphabet():
                self.assertIn(i, state.output_fun)

    def test_moore_structure(self):
        random.seed(3)
        moore = generate_random_deterministic_automata('moore', num_states=4, input_alphabet_size=2,
                                                        output_alphabet_size=3)
        self.assertIsInstance(moore, MooreMachine)
        for state in moore.states:
            self.assertIsNotNone(state.output)

    def test_invalid_automaton_type_raises(self):
        with self.assertRaises(AssertionError):
            generate_random_deterministic_automata('not_a_type', num_states=3, input_alphabet_size=2)

    def test_custom_input_alphabet_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            generate_random_deterministic_automata('dfa', num_states=3, input_alphabet_size=2,
                                                   custom_input_alphabet=['a', 'b', 'c'])

    def test_custom_output_alphabet_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            generate_random_deterministic_automata('mealy', num_states=3, input_alphabet_size=2,
                                                   output_alphabet_size=2, custom_output_alphabet=['a'])

    def test_num_accepting_states_respected(self):
        random.seed(4)
        dfa = generate_random_deterministic_automata('dfa', num_states=6, input_alphabet_size=2,
                                                      ensure_minimality=False, num_accepting_states=2)
        num_accepting = sum(1 for s in dfa.states if s.is_accepting)
        self.assertEqual(num_accepting, 2)

    def test_ensure_minimality_false_still_valid_automaton(self):
        random.seed(5)
        dfa = generate_random_deterministic_automata('dfa', num_states=4, input_alphabet_size=2,
                                                      ensure_minimality=False)
        self.assertEqual(len(dfa.states), 4)
        self.assertTrue(all_states_reachable(dfa))

    def test_ensure_minimality_true_produces_minimal_automaton(self):
        random.seed(6)
        for _ in range(5):
            dfa = generate_random_deterministic_automata('dfa', num_states=4, input_alphabet_size=2)
            self.assertTrue(dfa.is_minimal())
            self.assertEqual(dfa.size, 4)


class TestGenerateRandomDfaMealyMoore(unittest.TestCase):
    def test_generate_random_dfa(self):
        random.seed(7)
        dfa = generate_random_dfa(num_states=5, alphabet=['a', 'b'], num_accepting_states=2)
        self.assertEqual(len(dfa.states), 5)
        self.assertEqual(set(dfa.get_input_alphabet()), {'a', 'b'})

    def test_generate_random_dfa_too_many_accepting_states_is_corrected(self):
        random.seed(8)
        dfa = generate_random_dfa(num_states=4, alphabet=['a', 'b'], num_accepting_states=10,
                                  ensure_minimality=False)
        num_accepting = sum(1 for s in dfa.states if s.is_accepting)
        self.assertLessEqual(num_accepting, 4)

    def test_generate_random_mealy_machine(self):
        random.seed(9)
        mealy = generate_random_mealy_machine(num_states=4, input_alphabet=['a', 'b'],
                                              output_alphabet=['x', 'y', 'z'])
        self.assertIsInstance(mealy, MealyMachine)
        self.assertEqual(set(mealy.get_input_alphabet()), {'a', 'b'})

    def test_generate_random_moore_machine(self):
        random.seed(10)
        moore = generate_random_moore_machine(num_states=4, input_alphabet=['a', 'b'],
                                              output_alphabet=['x', 'y', 'z'])
        self.assertIsInstance(moore, MooreMachine)
        self.assertEqual(set(moore.get_input_alphabet()), {'a', 'b'})

    def test_compute_prefixes_flag(self):
        random.seed(11)
        mealy = generate_random_mealy_machine(num_states=3, input_alphabet=['a', 'b'],
                                              output_alphabet=['x', 'y'], compute_prefixes=True)
        for state in mealy.states:
            self.assertIsNotNone(state.prefix)


class TestGenerateRandomMdpSmm(unittest.TestCase):
    def test_generate_random_mdp_structure(self):
        random.seed(12)
        mdp = generate_random_mdp(num_states=5, input_size=2, output_size=3)
        self.assertIsInstance(mdp, Mdp)
        self.assertEqual(len(mdp.states), 5)
        for state in mdp.states:
            for i in mdp.get_input_alphabet():
                probs = [p for _, p in state.transitions[i]]
                self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_generate_random_mdp_deterministic_labeling(self):
        random.seed(13)
        mdp = generate_random_mdp(num_states=6, input_size=2, output_size=4)
        for state in mdp.states:
            for i in mdp.get_input_alphabet():
                outputs = [s.output for s, _ in state.transitions[i]]
                self.assertEqual(len(outputs), len(set(outputs)))

    def test_generate_random_smm_structure(self):
        random.seed(14)
        smm = generate_random_smm(num_states=5, input_size=2, output_size=3)
        self.assertIsInstance(smm, StochasticMealyMachine)
        for state in smm.states:
            for i in smm.get_input_alphabet():
                probs = [p for _, _, p in state.transitions[i]]
                self.assertAlmostEqual(sum(probs), 1.0, places=5)


class TestGenerateRandomOnfsm(unittest.TestCase):
    def test_structure(self):
        random.seed(15)
        onfsm = generate_random_ONFSM(num_states=5, num_inputs=3, num_outputs=3)
        self.assertIsInstance(onfsm, Onfsm)
        self.assertEqual(len(onfsm.states), 5)
        for state in onfsm.states:
            self.assertEqual(set(state.transitions.keys()), set(onfsm.get_input_alphabet()))

    def test_multiple_out_prob_zero_gives_single_output_per_transition(self):
        random.seed(16)
        onfsm = generate_random_ONFSM(num_states=4, num_inputs=2, num_outputs=3, multiple_out_prob=0.0)
        for state in onfsm.states:
            for i in state.transitions:
                self.assertEqual(len(state.transitions[i]), 1)


class TestGenerateRandomMarkovChain(unittest.TestCase):
    def test_structure(self):
        random.seed(17)
        mc = generate_random_markov_chain(num_states=5)
        self.assertIsInstance(mc, MarkovChain)
        self.assertEqual(len(mc.states), 5)

    def test_too_few_states_raises(self):
        with self.assertRaises(AssertionError):
            generate_random_markov_chain(num_states=2)

    def test_transition_probabilities_sum_to_one(self):
        random.seed(18)
        mc = generate_random_markov_chain(num_states=6)
        for state in mc.states[:-1]:
            probs = [p for _, p in state.transitions]
            self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_last_state_has_no_outgoing_transitions(self):
        random.seed(19)
        mc = generate_random_markov_chain(num_states=5)
        self.assertEqual(mc.states[-1].transitions, [])


class TestGenerateRandomSevpa(unittest.TestCase):
    def test_structure(self):
        random.seed(20)
        sevpa = generate_random_sevpa(num_states=5, internal_alphabet_size=2, call_alphabet_size=2,
                                      return_alphabet_size=2, acceptance_prob=0.5, return_transition_prob=0.5)
        self.assertIsInstance(sevpa, Sevpa)
        self.assertEqual(len(sevpa.states), 5)

    def test_all_internal_letters_defined_for_all_states(self):
        # regression test: the completeness pass used to check `transitions[letter] is None`, but
        # transitions is a defaultdict(list), so missing letters resolve to `[]`, never `None` -
        # meaning some states could be left without a transition on some internal letters.
        random.seed(21)
        sevpa = generate_random_sevpa(num_states=4, internal_alphabet_size=2, call_alphabet_size=2,
                                      return_alphabet_size=2, acceptance_prob=0.3, return_transition_prob=0.3)
        for state in sevpa.states:
            for internal_letter in sevpa.input_alphabet.internal_alphabet:
                self.assertIsNotNone(state.transitions[internal_letter])
                self.assertGreater(len(state.transitions[internal_letter]), 0)

    def test_return_transitions_defined_for_all_stack_states(self):
        random.seed(22)
        sevpa = generate_random_sevpa(num_states=3, internal_alphabet_size=1, call_alphabet_size=2,
                                      return_alphabet_size=2, acceptance_prob=0.5, return_transition_prob=0.7)
        for state in sevpa.states:
            for return_letter in sevpa.input_alphabet.return_alphabet:
                self.assertIsNotNone(state.transitions[return_letter])
                self.assertGreaterEqual(len(state.transitions[return_letter]), len(sevpa.states) * len(sevpa.input_alphabet.call_alphabet))


if __name__ == '__main__':
    unittest.main()
