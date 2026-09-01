import unittest

from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
from aalpy.learning_algs.general_passive.Instrumentation import MergeViolationDebugger, ProgressReport
from aalpy.learning_algs.general_passive.GsmNode import GsmNode, TransitionInfo


class TestProgressReport(unittest.TestCase):
    def test_records_pta_size_and_merge_counts_for_a_small_run(self):
        data = [((), True), (('a',), True), (('b',), True)]
        instrumentation = ProgressReport(lvl=2)

        run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
               data_format='labeled_sequences', instrumentation=instrumentation)

        self.assertEqual(instrumentation.pta_size, 3)
        # 'a' and 'b' children both have the same (True) output and no further transitions, so they merge
        self.assertGreaterEqual(instrumentation.nr_merged_states_total, 1)
        self.assertIn('pta creation time', instrumentation.stats)
        self.assertIn('learning time', instrumentation.stats)
        self.assertIn('total time', instrumentation.stats)

    def test_lvl_zero_skips_detailed_tracking(self):
        instrumentation = ProgressReport(lvl=0)
        self.assertFalse(hasattr(instrumentation, 'log'))

        data = [((), True), (('a',), False)]
        # should not raise even though detailed tracking attributes are absent
        run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
               data_format='labeled_sequences', instrumentation=instrumentation)


class TestMergeViolationDebugger(unittest.TestCase):
    def build_matching_ground_truth_tree(self):
        # data [((), True), (('a',), True), (('b',), True)] is only ever consistent with a single-state
        # automaton that self-loops on 'a' and 'b'; the ground truth tree must reflect that so that the
        # actual merges GSM performs (root with 'a', root with 'b') are considered correct.
        root = GsmNode((None, True), None)
        root.transitions['a'][True] = TransitionInfo(root, 1, None, None)
        root.transitions['b'][True] = TransitionInfo(root, 1, None, None)
        return root

    def test_logs_correct_merges_and_promotions_against_ground_truth(self):
        ground_truth = self.build_matching_ground_truth_tree()
        debugger = MergeViolationDebugger(ground_truth)

        data = [((), True), (('a',), True), (('b',), True)]
        run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
               data_format='labeled_sequences', instrumentation=debugger)

        kinds = [entry[0] for entry in debugger.log]
        self.assertIn('promote', kinds)
        self.assertNotIn('wrong promote', kinds)
        self.assertNotIn('wrong merge', kinds)
        self.assertNotIn('broken merge', kinds)

    def test_flags_wrong_merge_against_mismatched_ground_truth(self):
        # a ground truth tree where 'a' and 'b' are distinct states never merges them;
        # comparing against it while the actual run does merge them should be flagged as wrong.
        mismatched_ground_truth = GsmNode((None, True), None)
        mismatched_ground_truth.add_trace([('a', True)])
        mismatched_ground_truth.add_trace([('b', True)])
        # sabotage: make root.get_by_prefix for 'b' point to a node distinct from 'a's, but give it a
        # different (non-tree) identity so the debugger's identity check for a real merge fails
        debugger = MergeViolationDebugger(mismatched_ground_truth)

        # data where 'a' and 'b' children both lead to identical (mergeable) leaves
        data = [((), True), (('a',), True), (('b',), True)]
        run_GSM(data, output_behavior='moore', transition_behavior='deterministic',
               data_format='labeled_sequences', instrumentation=debugger)

        kinds = [entry[0] for entry in debugger.log]
        # since 'a' and 'b' are genuinely distinct nodes in the ground truth tree, merging them in the
        # actual run must be flagged as a wrong merge.
        self.assertIn('wrong merge', kinds)


if __name__ == '__main__':
    unittest.main()
