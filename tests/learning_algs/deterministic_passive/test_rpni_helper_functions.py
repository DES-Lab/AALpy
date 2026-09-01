import unittest

from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import (
    RpniNode, check_sequence, createPTA, extract_unique_sequences, to_automaton,
)


class TestRpniNode(unittest.TestCase):
    def test_default_moore_output_is_none(self):
        node = RpniNode(automaton_type='moore')
        self.assertIsNone(node.output)
        self.assertEqual(node.children, {})

    def test_default_mealy_output_is_empty_dict(self):
        node = RpniNode(automaton_type='mealy')
        self.assertEqual(node.output, {})

    def test_shallow_copy_shares_child_nodes_but_new_children_dict(self):
        child = RpniNode(automaton_type='moore')
        parent = RpniNode(output=True, children={'a': child}, automaton_type='moore')
        copy = parent.shallow_copy()
        self.assertIsNot(copy.children, parent.children)
        self.assertIs(copy.children['a'], child)

    def test_shallow_copy_mealy_output_is_independent_dict(self):
        parent = RpniNode(output={'a': 'x'}, automaton_type='mealy')
        copy = parent.shallow_copy()
        copy.output['b'] = 'y'
        self.assertNotIn('b', parent.output)

    def test_deep_copy_duplicates_subtree(self):
        child = RpniNode(automaton_type='moore')
        child.prefix = ('a',)
        parent = RpniNode(output=True, children={'a': child}, automaton_type='moore')
        parent.prefix = ()
        copy = parent.copy()
        self.assertIsNot(copy.children['a'], child)
        self.assertEqual(copy.children['a'].prefix, ('a',))

    def test_lt_compares_prefix_length(self):
        short = RpniNode(automaton_type='moore')
        short.prefix = ('a',)
        long = RpniNode(automaton_type='moore')
        long.prefix = ('a', 'b')
        self.assertTrue(short < long)
        self.assertFalse(long < short)

    def test_eq_compares_prefix_not_identity(self):
        n1 = RpniNode(automaton_type='moore')
        n1.prefix = ('a',)
        n2 = RpniNode(automaton_type='moore')
        n2.prefix = ('a',)
        self.assertEqual(n1, n2)
        self.assertIsNot(n1, n2)

    def test_compatible_outputs_moore_none_matches_anything(self):
        n1 = RpniNode(output=None, automaton_type='moore')
        n2 = RpniNode(output=True, automaton_type='moore')
        self.assertTrue(n1.compatible_outputs(n2))
        self.assertTrue(n2.compatible_outputs(n1))

    def test_compatible_outputs_moore_conflicting_values(self):
        n1 = RpniNode(output=True, automaton_type='moore')
        n2 = RpniNode(output=False, automaton_type='moore')
        self.assertFalse(n1.compatible_outputs(n2))

    def test_compatible_outputs_mealy_disjoint_inputs_are_compatible(self):
        n1 = RpniNode(output={'a': 'x'}, automaton_type='mealy')
        n2 = RpniNode(output={'b': 'y'}, automaton_type='mealy')
        self.assertTrue(n1.compatible_outputs(n2))

    def test_compatible_outputs_mealy_conflicting_shared_input(self):
        n1 = RpniNode(output={'a': 'x'}, automaton_type='mealy')
        n2 = RpniNode(output={'a': 'z'}, automaton_type='mealy')
        self.assertFalse(n1.compatible_outputs(n2))

    def test_get_child_by_prefix_follows_transitions(self):
        leaf = RpniNode(automaton_type='moore')
        mid = RpniNode(children={'b': leaf}, automaton_type='moore')
        root = RpniNode(children={'a': mid}, automaton_type='moore')
        self.assertIs(root.get_child_by_prefix(('a', 'b')), leaf)

    def test_get_child_by_prefix_empty_returns_self(self):
        root = RpniNode(automaton_type='moore')
        self.assertIs(root.get_child_by_prefix(()), root)


class TestCreatePTA(unittest.TestCase):
    def test_moore_builds_tree_with_correct_outputs(self):
        data = [((), True), (('a',), False), (('a', 'a'), True)]
        root = createPTA(data, 'moore')
        self.assertEqual(root.output, True)
        self.assertEqual(root.children['a'].output, False)
        self.assertEqual(root.children['a'].children['a'].output, True)

    def test_moore_conflicting_labels_returns_none(self):
        data = [((), True), ((), False)]
        self.assertIsNone(createPTA(data, 'moore'))

    def test_moore_conflicting_labels_at_leaf_returns_none(self):
        data = [(('a',), True), (('a',), False)]
        self.assertIsNone(createPTA(data, 'moore'))

    def test_mealy_builds_tree_with_transition_outputs(self):
        data = [(('a',), 'x'), (('a', 'b'), 'y')]
        root = createPTA(data, 'mealy')
        self.assertEqual(root.output, {'a': 'x'})
        self.assertEqual(root.children['a'].output, {'b': 'y'})

    def test_mealy_conflicting_labels_returns_none(self):
        data = [(('a',), 'x'), (('a',), 'z')]
        self.assertIsNone(createPTA(data, 'mealy'))

    def test_prefixes_are_recorded_on_nodes(self):
        data = [(('a', 'b'), True)]
        root = createPTA(data, 'dfa')
        self.assertEqual(root.prefix, ())
        self.assertEqual(root.children['a'].prefix, ('a',))
        self.assertEqual(root.children['a'].children['b'].prefix, ('a', 'b'))


class TestCheckSequence(unittest.TestCase):
    def test_valid_moore_sequence_accepted(self):
        data = [((), True), (('a',), False)]
        root = createPTA(data, 'moore')
        self.assertTrue(check_sequence(root, [True, ('a', False)], 'moore'))

    def test_invalid_moore_sequence_rejected(self):
        data = [((), True), (('a',), False)]
        root = createPTA(data, 'moore')
        self.assertFalse(check_sequence(root, [True, ('a', True)], 'moore'))

    def test_none_output_in_test_sequence_is_ignored(self):
        data = [((), True), (('a',), False)]
        root = createPTA(data, 'moore')
        self.assertTrue(check_sequence(root, [None, ('a', None)], 'moore'))

    def test_valid_mealy_sequence_accepted(self):
        data = [(('a',), 'x'), (('a', 'b'), 'y')]
        root = createPTA(data, 'mealy')
        self.assertTrue(check_sequence(root, [('a', 'x'), ('b', 'y')], 'mealy'))

    def test_invalid_mealy_sequence_rejected(self):
        data = [(('a',), 'x'), (('a', 'b'), 'y')]
        root = createPTA(data, 'mealy')
        self.assertFalse(check_sequence(root, [('a', 'z')], 'mealy'))


class TestExtractUniqueSequences(unittest.TestCase):
    def test_extracts_one_sequence_per_leaf_moore(self):
        data = [((), True), (('a',), False), (('b',), False)]
        root = createPTA(data, 'moore')
        sequences = extract_unique_sequences(root, 'moore')
        self.assertEqual(len(sequences), 2)
        self.assertIn([True, ('a', False)], sequences)
        self.assertIn([True, ('b', False)], sequences)

    def test_extracted_sequences_round_trip_through_check_sequence(self):
        data = [((), True), (('a', 'b'), False)]
        root = createPTA(data, 'moore')
        sequences = extract_unique_sequences(root, 'moore')
        for seq in sequences:
            self.assertTrue(check_sequence(root, seq, 'moore'))


class TestToAutomaton(unittest.TestCase):
    def test_dfa_none_outputs_default_to_false(self):
        data = [(('a',), True)]
        root = createPTA(data, 'dfa')
        # root has no explicit label -> output stays None until to_automaton fixes it up
        red = [root, root.children['a']]
        dfa = to_automaton(red, 'dfa')
        self.assertFalse(dfa.initial_state.output)
        self.assertTrue(dfa.initial_state.transitions['a'].output)

    def test_moore_conversion_preserves_topology(self):
        data = [((), 1), (('a',), 2), (('a', 'a'), 3)]
        root = createPTA(data, 'moore')
        red = [root, root.children['a'], root.children['a'].children['a']]
        moore = to_automaton(red, 'moore')
        self.assertEqual(moore.initial_state.output, 1)
        s1 = moore.initial_state.transitions['a']
        self.assertEqual(s1.output, 2)
        s2 = s1.transitions['a']
        self.assertEqual(s2.output, 3)

    def test_mealy_conversion_sets_output_fun(self):
        data = [(('a',), 'x'), (('a', 'b'), 'y')]
        root = createPTA(data, 'mealy')
        node_a = root.children['a']
        red = [root, node_a, node_a.children['b']]
        mealy = to_automaton(red, 'mealy')
        self.assertEqual(mealy.initial_state.output_fun['a'], 'x')
        self.assertEqual(mealy.initial_state.transitions['a'].output_fun['b'], 'y')


if __name__ == '__main__':
    unittest.main()
