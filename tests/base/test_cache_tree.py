import unittest

from aalpy.base.CacheTree import CacheDict, CacheTree


class TestCacheTree(unittest.TestCase):
    def test_in_cache_on_empty_tree_returns_none(self):
        tree = CacheTree()
        self.assertIsNone(tree.in_cache(('a', 'b')))

    def test_add_and_retrieve(self):
        tree = CacheTree()
        tree.add_to_cache(('a', 'b'), (1, 2))
        self.assertEqual(tree.in_cache(('a', 'b')), (1, 2))

    def test_in_cache_returns_prefix_output(self):
        tree = CacheTree()
        tree.add_to_cache(('a', 'b', 'c'), (1, 2, 3))
        self.assertEqual(tree.in_cache(('a', 'b')), (1, 2))

    def test_in_cache_missing_suffix_returns_none(self):
        tree = CacheTree()
        tree.add_to_cache(('a',), (1,))
        self.assertIsNone(tree.in_cache(('a', 'b')))

    def test_in_cache_unknown_branch_returns_none(self):
        tree = CacheTree()
        tree.add_to_cache(('a',), (1,))
        self.assertIsNone(tree.in_cache(('b',)))

    def test_empty_sequence_is_cached(self):
        tree = CacheTree()
        self.assertEqual(tree.in_cache(()), ())

    def test_reset_clears_current_position(self):
        tree = CacheTree()
        tree.reset()
        tree.step_in_cache('a', 1)
        self.assertEqual(tree.inputs, ('a',))
        tree.reset()
        self.assertEqual(tree.inputs, ())
        self.assertIs(tree.curr_node, tree.root_node)

    def test_step_in_cache_none_sets_root_value(self):
        tree = CacheTree()
        tree.reset()
        tree.step_in_cache(None, 'root_output')
        self.assertEqual(tree.root_node.value, 'root_output')

    def test_consistent_repeated_insert_is_fine(self):
        tree = CacheTree()
        tree.add_to_cache(('a', 'b'), (1, 2))
        # inserting the exact same sequence again must not raise
        tree.add_to_cache(('a', 'b'), (1, 2))
        self.assertEqual(tree.in_cache(('a', 'b')), (1, 2))

    def test_non_determinism_raises_system_exit(self):
        tree = CacheTree()
        tree.add_to_cache(('a',), (1,))
        with self.assertRaises(SystemExit):
            tree.add_to_cache(('a',), (2,))

    def test_branching_inputs_are_independent(self):
        tree = CacheTree()
        tree.add_to_cache(('a',), (1,))
        tree.add_to_cache(('b',), (2,))
        self.assertEqual(tree.in_cache(('a',)), (1,))
        self.assertEqual(tree.in_cache(('b',)), (2,))


class TestCacheDict(unittest.TestCase):
    def test_in_cache_on_empty_dict_returns_none(self):
        cache = CacheDict()
        self.assertIsNone(cache.in_cache(('a', 'b')))

    def test_add_and_retrieve(self):
        cache = CacheDict()
        cache.add_to_cache(('a', 'b'), (1, 2))
        self.assertEqual(cache.in_cache(('a', 'b')), (1, 2))

    def test_in_cache_missing_returns_none(self):
        cache = CacheDict()
        cache.add_to_cache(('a',), (1,))
        self.assertIsNone(cache.in_cache(('a', 'b')))

    def test_non_determinism_raises_system_exit(self):
        cache = CacheDict()
        cache.reset()
        cache.step_in_cache('a', 1)
        cache.reset()
        with self.assertRaises(SystemExit):
            cache.step_in_cache('a', 2)

    def test_consistent_repeated_step_is_fine(self):
        cache = CacheDict()
        cache.reset()
        cache.step_in_cache('a', 1)
        cache.reset()
        # re-affirming the same input/output for the same accumulated path must not raise
        cache.step_in_cache('a', 1)


if __name__ == '__main__':
    unittest.main()
