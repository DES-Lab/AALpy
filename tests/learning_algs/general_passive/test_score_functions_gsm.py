import unittest

from aalpy.learning_algs.general_passive.GsmNode import GsmNode, TransitionInfo, unknown_output
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import (
    AIC_score, EDSM_frequency_score, EDSM_score, ScoreCalculation, ScoreCombinator, ScoreWithKTail,
    ScoreWithSinks, differential_info, hoeffding_compatibility, local_to_global_compatibility, lower_threshold,
    make_greedy, transform_score,
)


def node_with_counts(counts, prefix_access_pair=(None, unknown_output)):
    """Builds a node whose single input 'i' has the given {output: count} outgoing transitions."""
    node = GsmNode(prefix_access_pair, None)
    for out_sym, count in counts.items():
        target = GsmNode(('i', out_sym), node)
        node.transitions['i'][out_sym] = TransitionInfo(target, count, target, count)
    return node


class TestScoreCalculationDefaults(unittest.TestCase):
    def test_default_local_compatibility_always_true(self):
        sc = ScoreCalculation()
        self.assertTrue(sc.local_compatibility(GsmNode((None, None), None), GsmNode((None, None), None)))
        self.assertFalse(sc.has_local_compatibility())

    def test_default_score_function_always_true(self):
        sc = ScoreCalculation()
        self.assertTrue(sc.score_function({}))
        self.assertFalse(sc.has_score_function())

    def test_custom_functions_are_detected_as_overridden(self):
        sc = ScoreCalculation(local_compatibility=lambda a, b: False, score_function=lambda p: 42)
        self.assertTrue(sc.has_local_compatibility())
        self.assertTrue(sc.has_score_function())


class TestHoeffdingCompatibility(unittest.TestCase):
    def test_identical_distributions_are_compatible(self):
        a = node_with_counts({'x': 100, 'y': 100})
        b = node_with_counts({'x': 100, 'y': 100})
        compat = hoeffding_compatibility(0.05)
        self.assertTrue(compat(a, b))

    def test_very_different_distributions_are_incompatible(self):
        a = node_with_counts({'x': 1000, 'y': 0})
        b = node_with_counts({'x': 0, 'y': 1000})
        compat = hoeffding_compatibility(0.05)
        self.assertFalse(compat(a, b))

    def test_zero_total_count_is_ignored(self):
        a = node_with_counts({})
        b = node_with_counts({'x': 100})
        compat = hoeffding_compatibility(0.05)
        self.assertTrue(compat(a, b))

    def test_disjoint_inputs_are_compatible(self):
        a = GsmNode((None, None), None)
        a.transitions['i']['x'] = TransitionInfo(GsmNode(('i', 'x'), a), 100, GsmNode(('i', 'x'), a), 100)
        b = GsmNode((None, None), None)
        b.transitions['j']['y'] = TransitionInfo(GsmNode(('j', 'y'), b), 100, GsmNode(('j', 'y'), b), 100)
        compat = hoeffding_compatibility(0.05)
        self.assertTrue(compat(a, b))


class TestScoreWithKTail(unittest.TestCase):
    def test_beyond_depth_k_is_always_compatible(self):
        always_false = ScoreCalculation(local_compatibility=lambda a, b: False)
        wrapped = ScoreWithKTail(always_false, k=1)

        root = GsmNode((None, None), None)
        blue_shallow = GsmNode(('a', None), root)
        blue_shallow_child = GsmNode(('a', None), blue_shallow)

        wrapped.reset()
        # first call establishes the depth offset at blue_shallow's depth (1)
        self.assertFalse(wrapped.local_compatibility(root, blue_shallow))
        # a node one level deeper than the offset (depth 2) is beyond k=1 -> compatible regardless
        self.assertTrue(wrapped.local_compatibility(root, blue_shallow_child))

    def test_within_depth_k_delegates_to_wrapped_score(self):
        always_false = ScoreCalculation(local_compatibility=lambda a, b: False)
        wrapped = ScoreWithKTail(always_false, k=5)
        root = GsmNode((None, None), None)
        blue = GsmNode(('a', None), root)
        wrapped.reset()
        self.assertFalse(wrapped.local_compatibility(root, blue))


class TestScoreWithSinks(unittest.TestCase):
    def test_rejects_merge_between_sink_and_non_sink(self):
        always_true = ScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink)
        wrapped.reset()

        sink_node = GsmNode((None, 'sink'), None)
        normal_node = GsmNode((None, 'normal'), None)
        self.assertFalse(wrapped.local_compatibility(sink_node, normal_node))

    def test_allows_merge_between_two_sinks_by_default(self):
        always_true = ScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink)
        wrapped.reset()

        sink_a = GsmNode((None, 'sink'), None)
        sink_b = GsmNode((None, 'sink'), None)
        self.assertTrue(wrapped.local_compatibility(sink_a, sink_b))

    def test_rejects_merge_between_two_sinks_when_disallowed(self):
        always_true = ScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink, allow_sink_merge=False)
        wrapped.reset()

        sink_a = GsmNode((None, 'sink'), None)
        sink_b = GsmNode((None, 'sink'), None)
        self.assertFalse(wrapped.local_compatibility(sink_a, sink_b))

    def test_sink_check_only_applies_on_first_call(self):
        always_true = ScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink, allow_sink_merge=False)
        wrapped.reset()

        sink_a = GsmNode((None, 'sink'), None)
        normal = GsmNode((None, 'normal'), None)
        # consume the "first call" check with a compatible (non-sink) pair
        self.assertTrue(wrapped.local_compatibility(normal, normal))
        # subsequent calls skip the sink check entirely, so this doesn't get rejected
        self.assertTrue(wrapped.local_compatibility(sink_a, sink_a))


class TestScoreCombinator(unittest.TestCase):
    def test_default_aggregate_compatibility_commits_to_first_non_none(self):
        s1 = ScoreCalculation(local_compatibility=lambda a, b: None)
        s2 = ScoreCalculation(local_compatibility=lambda a, b: False)
        combined = ScoreCombinator([s1, s2])
        self.assertFalse(combined.local_compatibility(None, None))

    def test_default_aggregate_compatibility_true_when_all_none(self):
        s1 = ScoreCalculation(local_compatibility=lambda a, b: None)
        combined = ScoreCombinator([s1])
        self.assertTrue(combined.local_compatibility(None, None))

    def test_default_aggregate_score_collects_all_scores(self):
        s1 = ScoreCalculation(score_function=lambda p: 1)
        s2 = ScoreCalculation(score_function=lambda p: 2)
        combined = ScoreCombinator([s1, s2])
        self.assertEqual(combined.score_function({}), [1, 2])

    def test_reset_delegates_to_all_scores(self):
        calls = []

        class Tracking(ScoreCalculation):
            def reset(self):
                calls.append(id(self))

        s1, s2 = Tracking(), Tracking()
        combined = ScoreCombinator([s1, s2])
        combined.reset()
        self.assertEqual(len(calls), 2)


class TestLocalToGlobalCompatibility(unittest.TestCase):
    def test_true_when_all_local_checks_pass(self):
        fun = local_to_global_compatibility(lambda a, b: True)
        self.assertTrue(fun({'old': 'new'}))

    def test_false_when_any_local_check_fails(self):
        fun = local_to_global_compatibility(lambda a, b: a != 'bad_new')
        self.assertFalse(fun({'old': 'bad_new'}))


class TestDifferentialInfo(unittest.TestCase):
    def test_merging_identical_nodes_does_not_change_likelihood(self):
        # merging two structurally identical nodes into one should not change the log-likelihood,
        # but should reduce the number of parameters (fewer distinct transitions after the merge).
        old1 = node_with_counts({'x': 5, 'y': 5})
        old2 = node_with_counts({'x': 5, 'y': 5})
        merged = node_with_counts({'x': 10, 'y': 10})
        part = {old1: merged, old2: merged}
        llh_diff, param_diff = differential_info(part)
        self.assertAlmostEqual(llh_diff, 0.0)
        self.assertGreater(param_diff, 0)


class TestScoreTransforms(unittest.TestCase):
    def test_transform_score_on_plain_value(self):
        self.assertEqual(transform_score(5, lambda x: x * 2), 10)

    def test_transform_score_on_callable(self):
        fun = transform_score(lambda part: 5, lambda x: x * 2)
        self.assertEqual(fun({}), 10)

    def test_transform_score_on_score_calculation(self):
        # regression test: transform_score used to reassign score.score_function to a lambda that
        # referenced score.score_function again, causing infinite recursion on the first call.
        sc = ScoreCalculation(score_function=lambda part: 5)
        transformed = transform_score(sc, lambda x: x * 2)
        self.assertIs(transformed, sc)
        self.assertEqual(transformed.score_function({}), 10)

    def test_transform_score_on_score_calculation_can_be_applied_twice(self):
        sc = ScoreCalculation(score_function=lambda part: 5)
        transform_score(sc, lambda x: x * 2)
        transform_score(sc, lambda x: x + 1)
        self.assertEqual(sc.score_function({}), 11)

    def test_make_greedy_rejects_only_false(self):
        self.assertTrue(make_greedy(0))
        self.assertTrue(make_greedy('anything'))
        self.assertFalse(make_greedy(False))

    def test_lower_threshold_rejects_values_at_or_below_threshold(self):
        self.assertEqual(lower_threshold(5, 3), 5)
        self.assertFalse(lower_threshold(3, 3))
        self.assertFalse(lower_threshold(1, 3))


class TestBuiltinScoreFunctions(unittest.TestCase):
    def test_aic_score_rejects_partitions_below_threshold(self):
        score_fun = AIC_score(alpha=1000)
        old1 = node_with_counts({'x': 5})
        merged = node_with_counts({'x': 5})
        result = score_fun({old1: merged})
        self.assertFalse(result)

    def test_edsm_frequency_score_counts_contradicted_evidence(self):
        score_fun = EDSM_frequency_score(min_evidence=-1)
        old_node = node_with_counts({'x': 5})
        new_node = node_with_counts({'x': 10})  # count changed by the merge -> contradicted evidence
        result = score_fun({old_node: new_node})
        self.assertEqual(result, 5)

    def test_edsm_frequency_score_rejects_below_min_evidence(self):
        score_fun = EDSM_frequency_score(min_evidence=10)
        old_node = node_with_counts({'x': 5})
        new_node = node_with_counts({'x': 10})
        result = score_fun({old_node: new_node})
        self.assertFalse(result)

    def test_edsm_score_counts_merged_minus_partitions(self):
        score_fun = EDSM_score(min_evidence=-1)
        merged = node_with_counts({})
        part = {node_with_counts({}): merged, node_with_counts({}): merged, node_with_counts({}): 'other'}
        result = score_fun(part)
        # 3 original nodes map to 2 distinct partition representatives -> 3 - 2 = 1
        self.assertEqual(result, 1)

    def test_edsm_score_rejects_below_min_evidence(self):
        score_fun = EDSM_score(min_evidence=5)
        merged = node_with_counts({})
        part = {node_with_counts({}): merged, node_with_counts({}): merged}
        result = score_fun(part)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
