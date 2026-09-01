import unittest

from aalpy.learning_algs.general_passive.GsmNode import (
    GsmNode, TransitionInfo, detect_data_format, intersection_iterator, union_iterator, unknown_output,
)


class TestIterators(unittest.TestCase):
    def test_intersection_iterator_only_common_keys(self):
        a = {'x': 1, 'y': 2}
        b = {'y': 20, 'z': 30}
        result = list(intersection_iterator(a, b))
        self.assertEqual(result, [('y', 2, 20)])

    def test_union_iterator_uses_default_for_missing(self):
        a = {'x': 1}
        b = {'y': 2}
        result = sorted(union_iterator(a, b, default=-1))
        self.assertEqual(result, [('x', 1, -1), ('y', -1, 2)])


class TestDetectDataFormat(unittest.TestCase):
    def test_empty_data_defaults_to_io_traces(self):
        self.assertEqual(detect_data_format([]), 'io_traces')

    def test_gsm_node_is_tree_format(self):
        node = GsmNode((None, unknown_output), None)
        self.assertEqual(detect_data_format(node), 'tree')

    def test_labeled_sequences_detected(self):
        data = [(('a', 'b'), 1), (('a',), 2)]
        self.assertEqual(detect_data_format(data), 'labeled_sequences')

    def test_io_traces_detected(self):
        data = [[('a', 'x'), ('b', 'y')], [('a', 'x')]]
        self.assertEqual(detect_data_format(data), 'io_traces')

    def test_non_sequence_data_raises(self):
        with self.assertRaises(ValueError):
            detect_data_format([1, 2, 3])

    def test_ambiguous_short_traces_default_without_consistency_check(self):
        # a single 2-tuple could be a labeled_sequence or an io_trace of length 1;
        # without check_consistency, the format is decided as soon as unambiguous, using the first entry
        data = [(('a',), 1)]
        fmt = detect_data_format(data)
        self.assertIn(fmt, ('labeled_sequences', 'io_traces'))


class TestGsmNodeBasics(unittest.TestCase):
    def test_root_has_no_predecessor_and_zero_prefix_length(self):
        root = GsmNode((None, unknown_output), None)
        self.assertIsNone(root.predecessor)
        self.assertEqual(root.get_prefix_length(), 0)
        self.assertEqual(root.get_prefix(), [])

    def test_add_trace_builds_chain_with_correct_prefix(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x'), ('b', 'y')])
        node_a = root.transitions['a']['x'].target
        node_ab = node_a.transitions['b']['y'].target
        self.assertEqual(node_a.get_prefix_length(), 1)
        self.assertEqual(node_ab.get_prefix(), [('a', 'x'), ('b', 'y')])
        self.assertIs(node_ab.get_root(), root)

    def test_add_trace_increments_count_for_repeated_trace(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        root.add_trace([('a', 'x')])
        t_info = root.transitions['a']['x']
        self.assertEqual(t_info.count, 2)
        self.assertEqual(t_info.original_count, 2)

    def test_get_by_prefix_returns_none_for_undefined_path(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        self.assertIsNone(root.get_by_prefix([('b', 'y')]))

    def test_get_by_prefix_ignores_leading_none_input(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        node = root.get_by_prefix([(None, 'initial'), ('a', 'x')])
        self.assertIs(node, root.transitions['a']['x'].target)

    def test_get_all_nodes_includes_root_and_children(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x'), ('b', 'y')])
        nodes = root.get_all_nodes()
        self.assertEqual(len(nodes), 3)

    def test_is_tree_true_for_pta(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        root.add_trace([('b', 'y')])
        self.assertTrue(root.is_tree())

    def test_is_tree_false_when_node_shared(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        shared = root.transitions['a']['x'].target
        # manually introduce a shared target to simulate a merged (non-tree) structure
        root.transitions['b']['y'] = TransitionInfo(shared, 1, None, None)
        self.assertFalse(root.is_tree())

    def test_shallow_copy_shares_targets_but_independent_transitions_dict(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        copy = root.shallow_copy()
        self.assertIsNot(copy.transitions, root.transitions)
        self.assertIs(copy.transitions['a']['x'].target, root.transitions['a']['x'].target)
        copy.transitions['b']['y'] = TransitionInfo(copy, 1, None, None)
        self.assertNotIn('b', root.transitions)

    def test_make_input_complete_adds_self_loops_for_missing_inputs(self):
        root = GsmNode((None, 'root_out'), None)
        root.add_trace([('a', 'x')])
        # 'b' is used elsewhere in the tree but not from root
        node_a = root.transitions['a']['x'].target
        node_a.add_trace([('b', 'y')])
        missing = root.make_input_complete()
        self.assertIn((root, 'b', 'root_out'), missing)
        self.assertIs(root.transitions['b']['root_out'].target, root)


class TestGsmNodeOrderingAndOutputs(unittest.TestCase):
    def test_lt_orders_by_prefix_length_then_lexicographically(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 1), ('a', 1)])
        root.add_trace([('b', 1)])
        node_a = root.transitions['a'][1].target
        node_b = root.transitions['b'][1].target
        node_aa = node_a.transitions['a'][1].target
        self.assertTrue(node_a < node_aa)
        self.assertTrue(node_a < node_b)  # same length, 'a' < 'b'
        self.assertFalse(node_b < node_a)

    def test_resolve_unknown_prefix_output_only_updates_if_unknown(self):
        node = GsmNode(('a', unknown_output), None)
        node.resolve_unknown_prefix_output('resolved')
        self.assertEqual(node.get_prefix_output(), 'resolved')
        node.resolve_unknown_prefix_output('other')
        self.assertEqual(node.get_prefix_output(), 'resolved')

    def test_add_labeled_sequence_sets_prefix_output_on_final_node(self):
        root = GsmNode((None, unknown_output), None)
        root.add_labeled_sequence((('a', 'b'), 'label1'))
        # only the final step's transition dict key is resolved from unknown_output to the real label;
        # intermediate steps remain keyed by unknown_output.
        node = root.get_by_prefix([('a', unknown_output), ('b', 'label1')])
        self.assertIsNotNone(node)
        self.assertEqual(node.get_prefix_output(), 'label1')

    def test_add_labeled_sequence_raises_on_conflicting_label_for_same_sequence(self):
        root = GsmNode((None, unknown_output), None)
        root.add_labeled_sequence((('a',), 'out1'))
        with self.assertRaises(ValueError):
            root.add_labeled_sequence((('a',), 'out2'))

    def test_is_locally_deterministic_true_for_single_output_per_input(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        self.assertTrue(root.is_locally_deterministic())

    def test_is_locally_deterministic_false_for_two_outputs_same_input(self):
        root = GsmNode((None, unknown_output), None)
        root.transitions['a']['x'] = TransitionInfo(GsmNode(('a', 'x'), root), 1, None, None)
        root.transitions['a']['y'] = TransitionInfo(GsmNode(('a', 'y'), root), 1, None, None)
        self.assertFalse(root.is_locally_deterministic())
        self.assertFalse(root.is_deterministic())

    def test_deterministic_compatible_true_when_no_shared_inputs(self):
        n1 = GsmNode((None, unknown_output), None)
        n1.add_trace([('a', 'x')])
        n2 = GsmNode((None, unknown_output), None)
        n2.add_trace([('b', 'y')])
        self.assertTrue(n1.deterministic_compatible(n2))

    def test_deterministic_compatible_false_on_output_mismatch_for_shared_input(self):
        n1 = GsmNode((None, unknown_output), None)
        n1.add_trace([('a', 'x')])
        n2 = GsmNode((None, unknown_output), None)
        n2.add_trace([('a', 'y')])
        self.assertFalse(n1.deterministic_compatible(n2))

    def test_deterministic_compatible_true_when_unknown_output_present(self):
        n1 = GsmNode((None, unknown_output), None)
        n1.transitions['a'][unknown_output] = TransitionInfo(GsmNode(('a', unknown_output), n1), 1, None, None)
        n2 = GsmNode((None, unknown_output), None)
        n2.add_trace([('a', 'x')])
        self.assertTrue(n1.deterministic_compatible(n2))

    def test_is_moore_true_when_child_output_matches_transition_output(self):
        root = GsmNode((None, 'root_out'), None)
        root.add_trace([('a', 'child_out')])
        self.assertTrue(root.is_moore())

    def test_is_moore_false_when_child_output_mismatches(self):
        root = GsmNode((None, 'root_out'), None)
        child = GsmNode(('a', 'transition_out'), root)
        root.transitions['a']['transition_out'] = TransitionInfo(child, 1, None, None)
        child.prefix_access_pair = ('a', 'different_child_out')
        self.assertFalse(root.is_moore())

    def test_moore_compatible_true_for_matching_or_unknown_outputs(self):
        n1 = GsmNode(('a', 'x'), None)
        n2 = GsmNode(('a', 'x'), None)
        n3 = GsmNode(('a', unknown_output), None)
        self.assertTrue(n1.moore_compatible(n2))
        self.assertTrue(n1.moore_compatible(n3))

    def test_moore_compatible_false_for_conflicting_outputs(self):
        n1 = GsmNode(('a', 'x'), None)
        n2 = GsmNode(('a', 'y'), None)
        self.assertFalse(n1.moore_compatible(n2))

    def test_count_sums_transition_counts(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        root.add_trace([('a', 'x')])
        root.add_trace([('b', 'y')])
        self.assertEqual(root.count(), 3)

    def test_local_log_likelihood_contribution_zero_for_single_outcome(self):
        # a deterministic transition (single outcome for its input) contributes 0 to the log-likelihood,
        # since n*log(n) - n*log(n) == 0 regardless of count.
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        root.add_trace([('a', 'x')])
        self.assertAlmostEqual(root.local_log_likelihood_contribution(), 0.0)

    def test_local_log_likelihood_contribution_negative_for_split_outcomes(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        root.add_trace([('a', 'y')])
        self.assertLess(root.local_log_likelihood_contribution(), 0.0)


class TestGsmNodeCreatePTA(unittest.TestCase):
    def test_labeled_sequences_format(self):
        data = [(('a', 'b'), 1), (('a', 'c'), 2)]
        root = GsmNode.createPTA(data, output_behavior='moore', data_format='labeled_sequences')
        node_a = root.transitions['a'][unknown_output].target
        self.assertEqual(node_a.get_prefix_length(), 1)

    def test_io_traces_moore_uses_first_output_as_root_output(self):
        data = [[0, ('a', 1)], [0, ('a', 1)]]
        root = GsmNode.createPTA(data, output_behavior='moore', data_format='io_traces')
        self.assertEqual(root.get_prefix_output(), 0)

    def test_io_traces_mealy_has_no_root_output(self):
        data = [[('a', 'x')]]
        root = GsmNode.createPTA(data, output_behavior='mealy', data_format='io_traces')
        self.assertEqual(root.get_prefix_output(), unknown_output)

    def test_tree_format_passthrough_requires_tree_structure(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        result = GsmNode.createPTA(root, output_behavior='mealy', data_format='tree')
        self.assertIs(result, root)

    def test_tree_format_rejects_non_tree(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        shared = root.transitions['a']['x'].target
        root.transitions['b']['y'] = TransitionInfo(shared, 1, None, None)
        with self.assertRaises(ValueError):
            GsmNode.createPTA(root, output_behavior='mealy', data_format='tree')


class TestGsmNodeToAutomaton(unittest.TestCase):
    def test_to_automaton_deterministic_moore(self):
        root = GsmNode((None, 0), None)
        root.add_trace([('a', 1)])
        automaton = root.to_automaton('moore', 'deterministic')
        self.assertEqual(automaton.initial_state.output, 0)
        self.assertEqual(automaton.initial_state.transitions['a'].output, 1)

    def test_to_automaton_raises_on_non_moore_structure_when_moore_requested(self):
        root = GsmNode((None, 'root_out'), None)
        child = GsmNode(('a', 'transition_out'), root)
        root.transitions['a']['transition_out'] = TransitionInfo(child, 1, None, None)
        child.prefix_access_pair = ('a', 'different_output')
        with self.assertRaises(ValueError):
            root.to_automaton('moore', 'deterministic')

    def test_to_automaton_deterministic_mealy(self):
        root = GsmNode((None, unknown_output), None)
        root.add_trace([('a', 'x')])
        automaton = root.to_automaton('mealy', 'deterministic')
        self.assertEqual(automaton.initial_state.output_fun['a'], 'x')


if __name__ == '__main__':
    unittest.main()
