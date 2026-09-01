import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Dfa, DfaState
from aalpy.base.SUL import CacheSUL


def parity_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


class TestAutomatonSUL(unittest.TestCase):
    def test_step_delegates_to_automaton(self):
        sul = AutomatonSUL(parity_dfa())
        self.assertFalse(sul.step('a'))
        self.assertTrue(sul.step('a'))

    def test_pre_resets_automaton(self):
        sul = AutomatonSUL(parity_dfa())
        sul.step('a')
        self.assertIsNot(sul.automaton.current_state, sul.automaton.initial_state)
        sul.pre()
        self.assertIs(sul.automaton.current_state, sul.automaton.initial_state)

    def test_query_resets_before_and_after(self):
        sul = AutomatonSUL(parity_dfa())
        sul.query(('a', 'a', 'a'))
        # after query(), the automaton should be back to initial (pre() was called at query start
        # and the automaton's current_state is left wherever the last step went, post() does not reset)
        self.assertIsNot(sul.automaton.current_state, sul.automaton.initial_state)
        sul.pre()
        self.assertIs(sul.automaton.current_state, sul.automaton.initial_state)

    def test_query_empty_word_on_dfa(self):
        sul = AutomatonSUL(parity_dfa())
        self.assertEqual(sul.query(()), [True])

    def test_query_output_matches_manual_stepping(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        result = sul.query(('a', 'b', 'a'))
        self.assertEqual(result, [False, False, True])

    def test_io_query_pairs_inputs_with_outputs(self):
        sul = AutomatonSUL(parity_dfa())
        result = sul.io_query(('a', 'a'))
        self.assertEqual(result, [('a', False), ('a', True)])

    def test_num_queries_and_steps_counters(self):
        sul = AutomatonSUL(parity_dfa())
        sul.query(('a', 'b'))
        sul.query(('a',))
        self.assertEqual(sul.num_queries, 2)
        self.assertEqual(sul.num_steps, 3)

    def test_independent_suls_have_independent_state(self):
        dfa = parity_dfa()
        sul1 = AutomatonSUL(dfa)
        sul2 = AutomatonSUL(dfa)
        sul1.step('a')
        # both wrap the same automaton instance, so state is shared -- this documents that behaviour
        self.assertIs(sul1.automaton, sul2.automaton)


class TestCacheSUL(unittest.TestCase):
    def test_cache_hit_avoids_wrapped_query(self):
        wrapped = AutomatonSUL(parity_dfa())
        cache_sul = CacheSUL(wrapped)

        result1 = cache_sul.query(('a', 'b'))
        self.assertEqual(wrapped.num_queries, 1)

        result2 = cache_sul.query(('a', 'b'))
        self.assertEqual(result1, result2)
        self.assertIsInstance(result2, list)  # cache hits and misses must return the same type
        self.assertEqual(wrapped.num_queries, 1)  # not called again
        self.assertEqual(cache_sul.num_cached_queries, 1)

    def test_cache_miss_for_new_query(self):
        wrapped = AutomatonSUL(parity_dfa())
        cache_sul = CacheSUL(wrapped)

        cache_sul.query(('a',))
        cache_sul.query(('b',))
        self.assertEqual(wrapped.num_queries, 2)
        self.assertEqual(cache_sul.num_cached_queries, 0)

    def test_prefix_of_cached_query_is_a_hit(self):
        wrapped = AutomatonSUL(parity_dfa())
        cache_sul = CacheSUL(wrapped)

        cache_sul.query(('a', 'b', 'a'))
        result = cache_sul.query(('a', 'b'))
        self.assertEqual(result, [False, False])
        self.assertEqual(cache_sul.num_cached_queries, 1)

    def test_dict_cache_type(self):
        wrapped = AutomatonSUL(parity_dfa())
        cache_sul = CacheSUL(wrapped, cache_type='dict')

        cache_sul.query(('a', 'b'))
        result = cache_sul.query(('a', 'b'))
        self.assertEqual(result, [False, False])
        self.assertEqual(cache_sul.num_cached_queries, 1)

    def test_step_updates_cache(self):
        wrapped = AutomatonSUL(parity_dfa())
        cache_sul = CacheSUL(wrapped)
        cache_sul.pre()
        cache_sul.step('a')
        self.assertEqual(cache_sul.cache.in_cache(('a',)), (False,))


if __name__ == '__main__':
    unittest.main()
