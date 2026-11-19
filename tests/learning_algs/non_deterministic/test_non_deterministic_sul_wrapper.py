import unittest

from aalpy.base import SUL
from aalpy.learning_algs.non_deterministic.NonDeterministicSULWrapper import NonDeterministicSULWrapper
from aalpy.learning_algs.non_deterministic.TraceTree import TraceTree


class AlternatingNonDetSUL(SUL):
    """A tiny hand-written non-deterministic SUL: 'a' alternates between 'x' and 'y' outputs,
    'b' always returns 'z'. Deterministic alternation (instead of random.choice) keeps the tests
    reproducible while still exercising the wrapper's handling of multiple observed outputs."""

    def __init__(self):
        super().__init__()
        self.pre_calls = 0
        self.post_calls = 0
        self.a_step_counter = 0

    def pre(self):
        self.pre_calls += 1

    def post(self):
        self.post_calls += 1

    def step(self, letter):
        if letter == 'a':
            out = 'x' if self.a_step_counter % 2 == 0 else 'y'
            self.a_step_counter += 1
        else:
            out = 'z'
        return out


class TestConstruction(unittest.TestCase):
    def test_wraps_given_sul_and_creates_empty_cache(self):
        raw = AlternatingNonDetSUL()
        wrapper = NonDeterministicSULWrapper(raw)
        self.assertIs(wrapper.sul, raw)
        self.assertIsInstance(wrapper.cache, TraceTree)
        self.assertIsNone(wrapper.cache.curr_node)


class TestPreAndPost(unittest.TestCase):
    def test_pre_resets_cache_cursor_and_delegates_to_wrapped_sul(self):
        raw = AlternatingNonDetSUL()
        wrapper = NonDeterministicSULWrapper(raw)
        wrapper.pre()
        self.assertIs(wrapper.cache.curr_node, wrapper.cache.root_node)
        self.assertEqual(raw.pre_calls, 1)

    def test_post_delegates_to_wrapped_sul(self):
        raw = AlternatingNonDetSUL()
        wrapper = NonDeterministicSULWrapper(raw)
        wrapper.post()
        self.assertEqual(raw.post_calls, 1)


class TestStep(unittest.TestCase):
    def test_step_returns_wrapped_output(self):
        raw = AlternatingNonDetSUL()
        wrapper = NonDeterministicSULWrapper(raw)
        wrapper.pre()
        self.assertEqual(wrapper.step('a'), 'x')
        self.assertEqual(wrapper.step('a'), 'y')

    def test_step_records_input_output_pair_in_cache(self):
        raw = AlternatingNonDetSUL()
        wrapper = NonDeterministicSULWrapper(raw)
        wrapper.pre()
        wrapper.step('a')
        self.assertIsNotNone(wrapper.cache.get_to_node(('a',), ('x',)))

    def test_step_moves_cache_cursor_along_the_path(self):
        raw = AlternatingNonDetSUL()
        wrapper = NonDeterministicSULWrapper(raw)
        wrapper.pre()
        wrapper.step('a')
        node_after_first = wrapper.cache.curr_node
        wrapper.step('a')
        self.assertIs(wrapper.cache.curr_node.parent, node_after_first)


class TestQueryAccumulatesTracesInCache(unittest.TestCase):
    def test_repeated_queries_accumulate_both_branches_of_a_non_det_input(self):
        wrapper = NonDeterministicSULWrapper(AlternatingNonDetSUL())
        for _ in range(4):
            wrapper.query(('a',))
        traces = wrapper.cache.get_all_traces(((), ()), ('a',))
        self.assertCountEqual(traces, [('x',), ('y',)])

    def test_deterministic_input_only_ever_records_one_output(self):
        wrapper = NonDeterministicSULWrapper(AlternatingNonDetSUL())
        for _ in range(3):
            wrapper.query(('b',))
        traces = wrapper.cache.get_all_traces(((), ()), ('b',))
        self.assertEqual(traces, [('z',)])

    def test_query_uses_base_sul_bookkeeping_for_queries_and_steps(self):
        wrapper = NonDeterministicSULWrapper(AlternatingNonDetSUL())
        wrapper.query(('a', 'b'))
        wrapper.query(('a',))
        self.assertEqual(wrapper.num_queries, 2)
        self.assertEqual(wrapper.num_steps, 3)

    def test_query_calls_pre_and_post_on_wrapped_sul_each_time(self):
        raw = AlternatingNonDetSUL()
        wrapper = NonDeterministicSULWrapper(raw)
        wrapper.query(('a',))
        wrapper.query(('a',))
        self.assertEqual(raw.pre_calls, 2)
        self.assertEqual(raw.post_calls, 2)

    def test_frequency_counter_tracks_how_often_each_branch_was_sampled(self):
        wrapper = NonDeterministicSULWrapper(AlternatingNonDetSUL())
        for _ in range(6):
            wrapper.query(('a',))
        self.assertEqual(wrapper.cache.get_s_e_sampling_frequency(((), ()), ('a',)), 6)


if __name__ == '__main__':
    unittest.main()
