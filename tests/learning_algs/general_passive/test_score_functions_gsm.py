import unittest

from aalpy.learning_algs.general_passive.DataHandler import CountOnPTADataHandler
from aalpy.learning_algs.general_passive.GsmNode import GsmNode, unknown_output
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import (
    ScoreCalculation,
    AIC_score, EDSM_frequency_score, EDSM_score, SimpleScoreCalculation, ScoreCombinator, ScoreWithKTail,
    ScoreWithSinks, differential_info, hoeffding_compatibility, local_to_global_compatibility, lower_threshold,
    greedy_score, score_transformation, SpecialScores
)


def node_with_counts(counts, prefix_access_pair=(None, unknown_output)):
    """Builds a node whose single input 'i' has the given {output: count} outgoing transitions."""
    dh = CountOnPTADataHandler()
    node = GsmNode(prefix_access_pair, None, dh.init_data())
    for out_sym, count in counts.items():
        target = GsmNode(('i', out_sym), node, dh.init_data())
        node.transitions['i'][out_sym] = target
    node.data.transition_count['i'] = counts
    node.data.pta_count['i'] = counts
    return node


class TestScoreCalculationDefaults(unittest.TestCase):
    def test_default_local_compatibility_always_true(self):
        sc = SimpleScoreCalculation()
        self.assertTrue(sc.local_compatibility(GsmNode((None, None), None, None), GsmNode((None, None), None, None)))

    def test_default_score_function_always_true(self):
        sc = SimpleScoreCalculation()
        self.assertTrue(sc.score_function({}))
        self.assertFalse(sc.has_score_function())

    def test_custom_functions_are_detected_as_overridden(self):
        sc = SimpleScoreCalculation(local_compatibility=lambda a, b: False, score_function=lambda p: 42)
        self.assertTrue(sc.has_score_function())


class TestOverrideDetection(unittest.TestCase):
    def test_plain_subclass_reports_no_overrides(self):
        class Plain(ScoreCalculation):
            pass

        self.assertFalse(Plain().has_local_compatibility())
        self.assertFalse(Plain().has_score_function())

    def test_subclass_overriding_methods_is_detected(self):
        class Custom(ScoreCalculation):
            def local_compatibility(self, a, b):
                return True

            def score_function(self, part):
                return 1

        self.assertTrue(Custom().has_local_compatibility())
        self.assertTrue(Custom().has_score_function())


class TestScoreCombinatorAggregation(unittest.TestCase):
    def test_no_early_verdict_when_no_sub_score_has_one(self):
        # a combined early verdict must stay NoScore, otherwise the partitioning is never built
        comb = ScoreCombinator([SimpleScoreCalculation(), SimpleScoreCalculation()])
        node = node_with_counts({})
        self.assertIs(comb.initialize_merge(node, node, True), SpecialScores.NoScore)

    def test_single_rejecting_sub_score_rejects(self):
        aggregate = ScoreCombinator.default_aggregate_score
        self.assertIs(aggregate([1, SpecialScores.ImmediateReject]), SpecialScores.ImmediateReject)

    def test_single_undecided_sub_score_stays_undecided(self):
        aggregate = ScoreCombinator.default_aggregate_score
        self.assertIs(aggregate([SpecialScores.NoScore, SpecialScores.ImmediateAccept]), SpecialScores.NoScore)

    def test_plain_scores_are_collected_into_a_list(self):
        self.assertEqual(ScoreCombinator.default_aggregate_score([1, 2]), [1, 2])


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
        dh = CountOnPTADataHandler()
        a = GsmNode((None, None), None, dh.init_data())
        a.data.transition_count = a.data.pta_count = {'i': {'x': 100}}
        b = GsmNode((None, None), None, dh.init_data())
        b.data.transition_count = b.data.pta_count = {'j': {'y': 100}}
        compat = hoeffding_compatibility(0.05)
        self.assertTrue(compat(a, b))


class TestScoreWithKTail(unittest.TestCase):
    def test_beyond_depth_k_is_always_compatible(self):
        always_false = SimpleScoreCalculation(local_compatibility=lambda a, b: False)
        wrapped = ScoreWithKTail(always_false, k=1)

        root = GsmNode((None, None), None, None)
        blue_shallow = GsmNode(('a', None), root, None)
        blue_shallow_child = GsmNode(('a', None), blue_shallow, None)

        wrapped.initialize_merge(root, blue_shallow, True)
        # first call establishes the depth offset at blue_shallow's depth (1)
        self.assertFalse(wrapped.local_compatibility(root, blue_shallow))
        # a node one level deeper than the offset (depth 2) is beyond k=1 -> compatible regardless
        self.assertIsNone(wrapped.local_compatibility(root, blue_shallow_child))

    def test_within_depth_k_delegates_to_wrapped_score(self):
        always_false = SimpleScoreCalculation(local_compatibility=lambda a, b: False)
        wrapped = ScoreWithKTail(always_false, k=5)
        root = GsmNode((None, None), None, None)
        blue = GsmNode(('a', None), root, None)
        wrapped.initialize_merge(root, blue, True)
        self.assertFalse(wrapped.local_compatibility(root, blue))


class TestScoreWithSinks(unittest.TestCase):
    def test_rejects_merge_between_sink_and_non_sink(self):
        always_true = SimpleScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink)

        sink_node = GsmNode((None, 'sink'), None, None)
        normal_node = GsmNode((None, 'normal'), None, None)
        # early reject
        self.assertFalse(wrapped.initialize_merge(sink_node, normal_node, True))
        # accept if encountered later
        self.assertTrue(wrapped.local_compatibility(sink_node, normal_node))

    def test_allows_merge_between_two_sinks_by_default(self):
        always_true = SimpleScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink)

        sink_a = GsmNode((None, 'sink'), None, None)
        sink_b = GsmNode((None, 'sink'), None, None)
        self.assertIsNone(wrapped.initialize_merge(sink_a, sink_b, True))
        self.assertTrue(wrapped.local_compatibility(sink_a, sink_b))

    def test_rejects_merge_between_two_sinks_when_disallowed(self):
        always_true = SimpleScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink, allow_sink_merge=False)

        sink_a = GsmNode((None, 'sink'), None, None)
        sink_b = GsmNode((None, 'sink'), None, None)
        self.assertFalse(wrapped.initialize_merge(sink_a, sink_b, True))
        self.assertTrue(wrapped.local_compatibility(sink_a, sink_b))

    def test_sink_check_only_applies_on_first_call(self):
        always_true = SimpleScoreCalculation(local_compatibility=lambda a, b: True)
        is_sink = lambda n: n.get_prefix_output() == 'sink'
        wrapped = ScoreWithSinks(always_true, sink_cond=is_sink, allow_sink_merge=False)

        sink_a = GsmNode((None, 'sink'), None, None)
        normal = GsmNode((None, 'normal'), None, None)
        self.assertFalse(wrapped.initialize_merge(normal, normal, True))
        # consume the "first call" check with a compatible (non-sink) pair
        self.assertTrue(wrapped.local_compatibility(normal, normal))
        # subsequent calls skip the sink check entirely, so this doesn't get rejected
        self.assertTrue(wrapped.local_compatibility(sink_a, sink_a))


class TestScoreCombinator(unittest.TestCase):
    def test_default_aggregate_compatibility_commits_to_first_non_none(self):
        s1 = SimpleScoreCalculation(local_compatibility=lambda a, b: None)
        s2 = SimpleScoreCalculation(local_compatibility=lambda a, b: False)
        combined = ScoreCombinator([s1, s2])
        self.assertFalse(combined.local_compatibility(None, None))

    def test_default_aggregate_compatibility_none_when_all_none(self):
        s1 = SimpleScoreCalculation(local_compatibility=lambda a, b: None)
        combined = ScoreCombinator([s1])
        self.assertIsNone(combined.local_compatibility(None, None))

    def test_default_aggregate_score_collects_all_scores(self):
        s1 = SimpleScoreCalculation(score_function=lambda p: 1)
        s2 = SimpleScoreCalculation(score_function=lambda p: 2)
        combined = ScoreCombinator([s1, s2])
        self.assertEqual(combined.score_function({}), [1, 2])

    def test_reset_delegates_to_all_scores(self):
        calls = []

        class Tracking(SimpleScoreCalculation):
            def initialize_merge(self, red, blue, first_pass):
                calls.append(id(self))

        s1, s2 = Tracking(), Tracking()
        combined = ScoreCombinator([s1, s2])
        combined.initialize_merge(None, None, True)
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
        self.assertEqual(score_transformation(lambda x: x * 2)(5), 10)

    def test_transform_score_on_callable(self):
        fun = score_transformation(lambda x: x * 2)(lambda part: 5)
        self.assertEqual(fun({}), 10)

    def test_transform_score_on_score_calculation(self):
        # regression test: transform_score used to reassign score.score_function to a lambda that
        # referenced score.score_function again, causing infinite recursion on the first call.
        sc = SimpleScoreCalculation(score_function=lambda part: 5)
        transformed = score_transformation(lambda x: x * 2)(sc)
        self.assertIs(transformed, sc)
        self.assertEqual(transformed.score_function({}), 10)

    def test_transform_score_on_score_calculation_can_be_applied_twice(self):
        sc = SimpleScoreCalculation(score_function=lambda part: 5)
        score_transformation(lambda x: x * 2)(sc)
        score_transformation(lambda x: x + 1)(sc)
        self.assertEqual(sc.score_function({}), 11)

    def test_make_greedy_rejects_only_false(self):
        self.assertTrue(greedy_score(0))
        self.assertTrue(greedy_score('anything') is SpecialScores.ImmediateAccept)
        self.assertTrue(greedy_score(False) is SpecialScores.ImmediateReject)

    def test_lower_threshold_rejects_values_at_or_below_threshold(self):
        self.assertEqual(lower_threshold(5, 3), 5)
        self.assertEqual(lower_threshold(3, 3), 3)
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
