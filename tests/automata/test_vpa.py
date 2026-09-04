import random
import unittest

from aalpy.automata import Vpa, VpaAlphabet, VpaState, VpaTransition


def balanced_parens_vpa():
    """
    Single-state VPA recognizing the Dyck language over '(' (push) / ')' (pop).
    q0 is initial and accepting; accepted iff the stack is empty.
    """
    q0 = VpaState('q0', is_accepting=True)
    q0.transitions['('].append(VpaTransition(q0, q0, '(', 'push', '('))
    q0.transitions[')'].append(VpaTransition(q0, q0, ')', 'pop', '('))
    return Vpa(q0, [q0]), q0


class TestVpaState(unittest.TestCase):
    def test_default_transitions_is_empty_defaultdict(self):
        state = VpaState('s')
        self.assertEqual(state.transitions['unused_key'], [])

    def test_default_not_accepting(self):
        state = VpaState('s')
        self.assertFalse(state.is_accepting)


class TestVpaStep(unittest.TestCase):
    def test_accepts_empty_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        self.assertTrue(vpa.step(None))

    def test_accepts_balanced_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')
        result = vpa.step(')')
        self.assertTrue(result)
        self.assertEqual(vpa.stack, [])

    def test_rejects_incomplete_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        result = vpa.step('(')
        self.assertFalse(result)
        self.assertEqual(vpa.stack, ['('])

    def test_unmatched_return_symbol_traps_in_error_state(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        result = vpa.step(')')
        self.assertFalse(result)
        self.assertIs(vpa.current_state, Vpa.error_state)

        # once trapped, further steps stay False without raising
        self.assertFalse(vpa.step('('))
        self.assertFalse(vpa.step(None))

    def test_nested_balanced_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        outputs = [vpa.step(c) for c in '(())']
        self.assertEqual(outputs, [False, False, False, True])

    def test_top_of_empty_stack_is_empty_list(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        self.assertEqual(vpa.top(), [])

    def test_top_reflects_last_pushed_symbol(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')
        self.assertEqual(vpa.top(), '(')


class TestVpaExecuteSequence(unittest.TestCase):
    def test_execute_sequence_matches_stepwise(self):
        vpa, q0 = balanced_parens_vpa()
        result = vpa.execute_sequence(q0, ['(', '(', ')', ')'], stack=[])
        self.assertEqual(result, [False, False, False, True])
        self.assertEqual(vpa.stack, [])

    def test_execute_sequence_empty_returns_empty_list(self):
        vpa, q0 = balanced_parens_vpa()
        self.assertEqual(vpa.execute_sequence(q0, [], stack=[]), [])

    def test_execute_sequence_ignores_leftover_stack_from_prior_use(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')  # leaves the stack non-empty: ['(']
        self.assertEqual(vpa.stack, ['('])

        result = vpa.execute_sequence(q0, [')'], stack=[])
        # a stale stack would make ')' match and incorrectly report acceptance;
        # execute_sequence must start from the given stack, not whatever was left over
        self.assertEqual(result, [False])

    def test_execute_sequence_resumes_from_explicit_stack(self):
        vpa, q0 = balanced_parens_vpa()
        result = vpa.execute_sequence(q0, [')'], stack=['('])
        # resuming as if one '(' had already been pushed makes the lone ')' balance out
        self.assertEqual(result, [True])
        self.assertEqual(vpa.stack, [])

    def test_execute_sequence_with_explicit_stack_matches_manual_stepping(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')
        vpa.step('(')
        stack_after_two_pushes = list(vpa.stack)
        manual_result = [vpa.step(c) for c in '))']

        result = vpa.execute_sequence(q0, [')', ')'], stack=stack_after_two_pushes)
        self.assertEqual(result, manual_result)


class TestVpaStructural(unittest.TestCase):
    def test_get_input_alphabet(self):
        vpa, q0 = balanced_parens_vpa()
        alphabet = vpa.get_input_alphabet()
        self.assertEqual(alphabet.call_alphabet, ['('])
        self.assertEqual(alphabet.return_alphabet, [')'])
        self.assertEqual(alphabet.internal_alphabet, [])

    def test_get_merged_alphabet(self):
        alphabet = VpaAlphabet(internal_alphabet=['i'], call_alphabet=['c'], return_alphabet=['r'])
        self.assertEqual(alphabet.get_merged_alphabet(), ['i', 'c', 'r'])

    def test_is_input_complete_true(self):
        vpa, q0 = balanced_parens_vpa()
        self.assertTrue(vpa.is_input_complete())

    def test_is_input_complete_false(self):
        # the alphabet is inferred from ALL transitions present in the automaton, so we need a second
        # state that does use ')' for the alphabet to include it, exposing q0's missing transition
        q0 = VpaState('q0', is_accepting=False)
        q1 = VpaState('q1', is_accepting=True)
        q0.transitions['('].append(VpaTransition(q0, q1, '(', 'push', '('))
        q1.transitions[')'].append(VpaTransition(q1, q1, ')', 'pop', '('))
        # q0 has no ')' transition
        vpa = Vpa(q0, [q0, q1])
        self.assertFalse(vpa.is_input_complete())


class TestVpaStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        vpa, q0 = balanced_parens_vpa()
        setup = vpa.to_state_setup()
        rebuilt = Vpa.from_state_setup(setup, init_state_id='q0')

        for w in [[], ['('], ['(', ')'], ['(', '(', ')', ')']]:
            rebuilt.reset_to_initial()
            outputs = [rebuilt.step(letter) for letter in w]
            vpa.reset_to_initial()
            expected = [vpa.step(letter) for letter in w]
            self.assertEqual(outputs, expected)


class TestVpaRandomAcceptingWord(unittest.TestCase):
    def test_generate_random_accepting_word_is_actually_accepting(self):
        vpa, q0 = balanced_parens_vpa()
        random.seed(0)
        # the walk is randomized and may run out of steps before balancing; a generous
        # max_steps budget keeps this reliable without pinning the RNG's exact trajectory
        word = vpa.generate_random_accepting_word(min_steps=2, max_steps=200)
        self.assertIsNotNone(word)

        vpa.reset_to_initial()
        outputs = [vpa.step(letter) for letter in word]
        self.assertTrue(outputs[-1])
        self.assertEqual(vpa.stack, [])


def redundant_dyck_vpa():
    """
    Dyck language over '(' / ')' with two interchangeable copies of the state reached by an internal symbol.
    The copies give well-matched words distinct transductions while leaving them in the same congruence class.
    """
    q0 = VpaState('q0', is_accepting=True)
    qa = VpaState('qa', is_accepting=True)
    qb = VpaState('qb', is_accepting=True)
    for state in (q0, qa, qb):
        state.transitions['('].append(VpaTransition(state, state, '(', 'push', '('))
        state.transitions[')'].append(VpaTransition(state, state, ')', 'pop', '('))
    for state in (q0, qa, qb):
        state.transitions['i'].append(VpaTransition(state, qa, 'i', None, None))
        state.transitions['j'].append(VpaTransition(state, qb, 'j', None, None))
    return Vpa(q0, [q0, qa, qb])


def brute_force_separating_context(vpa, word_1, word_2, max_left, max_right):
    """Searches all contexts up to the given lengths for one telling the two well-matched words apart."""
    from itertools import product

    from aalpy.automata.Vpa import vpa_context_separates_words

    alphabet = vpa.get_input_alphabet().get_merged_alphabet()
    for left_length in range(max_left + 1):
        for left in product(alphabet, repeat=left_length):
            for right_length in range(max_right + 1):
                for right in product(alphabet, repeat=right_length):
                    if vpa_context_separates_words(vpa, word_1, word_2, (left, right)):
                        return left, right
    return None


def congruence_class_representatives(vpa):
    """Maps the transduction of every class of the transition cover to a witness word of that class."""
    from aalpy.automata.Vpa import vpa_transition_cover, vpa_word_transduction

    representatives = {}
    for word in vpa_transition_cover(vpa):
        representatives.setdefault(vpa_word_transduction(vpa, word), word)
    return representatives


class TestComputeCharacterizingSet(unittest.TestCase):
    """
    The characterizing set follows the congruence of Alur et al., "Congruences for Visibly Pushdown Languages":
    it is a set of two-sided contexts telling apart the congruence classes of well-matched words. VpaCongruence
    decides that congruence exactly, so the set is characterizing rather than characterizing up to a search bound.
    """

    def characterizing_set_separates_all_congruence_classes(self, vpa):
        from aalpy.automata.Vpa import VpaCongruence, vpa_context_separates_words

        congruence = VpaCongruence(vpa)
        characterizing_set = vpa.compute_characterizing_set()
        representatives = list(congruence_class_representatives(vpa).items())

        for index, (transduction_1, word_1) in enumerate(representatives):
            for transduction_2, word_2 in representatives[index + 1:]:
                if congruence.separating_context(transduction_1, transduction_2) is None:
                    continue
                self.assertTrue(any(vpa_context_separates_words(vpa, word_1, word_2, c) for c in characterizing_set),
                                f'{word_1} and {word_2} are in different congruence classes but are not separated '
                                f'by the characterizing set')

    def test_separates_all_congruence_classes_of_balanced_parens(self):
        vpa, _ = balanced_parens_vpa()
        self.characterizing_set_separates_all_congruence_classes(vpa)

    def test_separates_all_congruence_classes_of_benchmark_vpas(self):
        from aalpy.utils.BenchmarkVpaModels import vpa_L1, vpa_L3, vpa_L9, vpa_for_even_parentheses

        for model in (vpa_L1(), vpa_L3(), vpa_L9(), vpa_for_even_parentheses()):
            self.characterizing_set_separates_all_congruence_classes(model)

    def test_every_reported_context_really_separates_the_pair_it_was_found_for(self):
        from aalpy.automata.Vpa import VpaCongruence, vpa_context_separates_words
        from aalpy.utils.BenchmarkVpaModels import vpa_L3, vpa_for_odd_parentheses

        for vpa in (vpa_L3(), vpa_for_odd_parentheses()):
            congruence = VpaCongruence(vpa)
            representatives = list(congruence_class_representatives(vpa).items())
            for index, (transduction_1, word_1) in enumerate(representatives):
                for transduction_2, word_2 in representatives[index + 1:]:
                    context = congruence.separating_context(transduction_1, transduction_2)
                    if context is not None:
                        self.assertTrue(vpa_context_separates_words(vpa, word_1, word_2, context))

    def test_classes_reported_as_congruent_survive_a_brute_force_search(self):
        """
        The decision procedure is exact, so a pair it reports as congruent must not be separated by any context.
        The redundant VPA accepts every balanced word, so all of its classes really are congruent, even though the
        interchangeable copies give them different transductions.
        """
        from aalpy.automata.Vpa import VpaCongruence

        vpa = redundant_dyck_vpa()
        congruence = VpaCongruence(vpa)
        representatives = list(congruence_class_representatives(vpa).items())
        self.assertGreater(len(representatives), 1, 'the VPA should have several transductions to compare')

        congruent_pairs = 0
        for index, (transduction_1, word_1) in enumerate(representatives):
            for transduction_2, word_2 in representatives[index + 1:]:
                self.assertIsNone(congruence.separating_context(transduction_1, transduction_2))
                congruent_pairs += 1
                self.assertIsNone(brute_force_separating_context(vpa, word_1, word_2, 3, 4),
                                  f'{word_1} and {word_2} were reported as congruent but are separable')

        self.assertGreater(congruent_pairs, 0)
        # a single congruence class needs no separating context, only the trivial one
        self.assertEqual([((), ())], vpa.compute_characterizing_set())

    def test_agrees_with_brute_force_on_both_answers(self):
        from aalpy.automata.Vpa import VpaCongruence

        vpa, _ = balanced_parens_vpa_with_two_classes()
        congruence = VpaCongruence(vpa)
        representatives = list(congruence_class_representatives(vpa).items())

        for index, (transduction_1, word_1) in enumerate(representatives):
            for transduction_2, word_2 in representatives[index + 1:]:
                decided = congruence.separating_context(transduction_1, transduction_2)
                brute_forced = brute_force_separating_context(vpa, word_1, word_2, 3, 4)
                self.assertEqual(decided is None, brute_forced is None,
                                 f'exact procedure and brute force disagree on {word_1} and {word_2}')

    def test_contexts_are_pairs_of_sequences(self):
        from aalpy.utils.BenchmarkVpaModels import vpa_L1

        for context in vpa_L1().compute_characterizing_set():
            self.assertEqual(2, len(context))
            self.assertIsInstance(context[0], tuple)
            self.assertIsInstance(context[1], tuple)

    def test_every_covered_word_is_well_matched(self):
        from aalpy.automata.Vpa import vpa_well_matched_cover
        from aalpy.utils import is_balanced
        from aalpy.utils.BenchmarkVpaModels import vpa_L3, vpa_for_even_parentheses

        for model in (vpa_L3(), vpa_for_even_parentheses()):
            for word in vpa_well_matched_cover(model).values():
                self.assertTrue(is_balanced(list(word), model.get_input_alphabet()))

    def test_plugging_a_covered_word_into_a_context_stays_well_matched(self):
        from aalpy.automata.Vpa import vpa_transition_cover
        from aalpy.utils import is_balanced
        from aalpy.utils.BenchmarkVpaModels import vpa_for_even_parentheses

        vpa = vpa_for_even_parentheses()
        alphabet = vpa.get_input_alphabet()
        for left, right in vpa.compute_characterizing_set():
            for word in vpa_transition_cover(vpa):
                self.assertTrue(is_balanced(list(left + word + right), alphabet))


def balanced_parens_vpa_with_two_classes():
    """Balanced parentheses that additionally have to end with an internal 'i', giving two congruence classes."""
    q0 = VpaState('q0', is_accepting=False)
    q1 = VpaState('q1', is_accepting=True)
    for state in (q0, q1):
        state.transitions['('].append(VpaTransition(state, q0, '(', 'push', '('))
        state.transitions[')'].append(VpaTransition(state, q0, ')', 'pop', '('))
        state.transitions['i'].append(VpaTransition(state, q1, 'i', None, None))
    return Vpa(q0, [q0, q1]), q0


if __name__ == '__main__':
    unittest.main()
