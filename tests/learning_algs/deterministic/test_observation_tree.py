import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs.deterministic.ObservationTree import MealyNode, MooreNode, ObservationTree
from aalpy.utils import get_Angluin_dfa


def dfa_tree(extension_rule=None, separation_rule='SepSeq'):
    sul = AutomatonSUL(get_Angluin_dfa())
    return ObservationTree(['a', 'b'], sul, 'dfa', extension_rule, separation_rule)


def mealy_tree(extension_rule=None, separation_rule='SepSeq'):
    sul = AutomatonSUL(get_Angluin_dfa())
    return ObservationTree(['a', 'b'], sul, 'mealy', extension_rule, separation_rule)


class TestObservationTreeInit(unittest.TestCase):
    def test_dfa_root_output_is_queried_eagerly(self):
        tree = dfa_tree()
        self.assertIsInstance(tree.root, MooreNode)
        self.assertTrue(tree.root.output)

    def test_mealy_root_has_no_output_attribute_use(self):
        tree = mealy_tree()
        self.assertIsInstance(tree.root, MealyNode)

    def test_basis_starts_with_only_the_root(self):
        tree = dfa_tree()
        self.assertEqual(tree.basis, [tree.root])
        self.assertEqual(tree.frontier_to_basis_dict, {})


class TestInsertAndGetObservation(unittest.TestCase):
    def test_insert_and_retrieve_matches(self):
        tree = dfa_tree()
        tree.insert_observation(['a', 'b'], [False, True])
        self.assertEqual(tree.get_observation(['a', 'b']), [False, True])
        self.assertEqual(tree.get_observation(['a']), [False])

    def test_mismatched_lengths_raise(self):
        tree = dfa_tree()
        with self.assertRaises(ValueError):
            tree.insert_observation(['a', 'b'], [False])

    def test_unseen_sequence_returns_none(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        self.assertIsNone(tree.get_observation(['b']))
        self.assertIsNone(tree.get_observation(['a', 'b']))

    def test_mealy_reinserting_same_input_with_conflicting_output_raises(self):
        tree = mealy_tree()
        tree.insert_observation(['a'], [1])
        with self.assertRaises(Exception):
            tree.insert_observation(['a'], [2])

    def test_mealy_reinserting_same_input_output_is_idempotent(self):
        tree = mealy_tree()
        tree.insert_observation(['a'], [1])
        tree.insert_observation(['a'], [1])
        self.assertEqual(tree.get_observation(['a']), [1])


class TestGetOutputsMatchesGetObservation(unittest.TestCase):
    def test_get_outputs_from_root_agrees_with_get_observation(self):
        tree = dfa_tree()
        tree.insert_observation(['a', 'b'], [False, True])
        self.assertEqual(tree.get_outputs(tree.root, ['a', 'b']), tree.get_observation(['a', 'b']))

    def test_get_outputs_from_non_root_basis_state(self):
        tree = dfa_tree()
        tree.insert_observation(['a', 'b'], [False, True])
        node_a = tree.get_successor(['a'])
        self.assertEqual(tree.get_outputs(node_a, ['b']), [True])

    def test_get_outputs_mealy_from_root_agrees_with_get_observation(self):
        tree = mealy_tree()
        tree.insert_observation(['a', 'b'], [1, 2])
        self.assertEqual(tree.get_outputs(tree.root, ['a', 'b']), tree.get_observation(['a', 'b']))


class TestTraversalHelpers(unittest.TestCase):
    def test_get_successor_and_transfer_and_access_sequence(self):
        tree = dfa_tree()
        tree.insert_observation(['a', 'b'], [False, True])
        node_ab = tree.get_successor(['a', 'b'])
        self.assertIsNotNone(node_ab)
        self.assertEqual(tree.get_transfer_sequence(tree.root, node_ab), ['a', 'b'])
        self.assertEqual(tree.get_access_sequence(node_ab), ('a', 'b'))

    def test_get_successor_unknown_path_returns_none(self):
        tree = dfa_tree()
        self.assertIsNone(tree.get_successor(['a']))

    def test_get_transfer_sequence_returns_none_for_unrelated_node(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.insert_observation(['b'], [False])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        # node_a is not an ancestor of node_b, so there is no transfer sequence between them.
        self.assertIsNone(tree.get_transfer_sequence(node_a, node_b))

    def test_get_size_counts_all_created_nodes(self):
        tree = dfa_tree()
        size_before = tree.get_size()
        tree.insert_observation(['a', 'b'], [False, True])
        self.assertEqual(tree.get_size(), size_before + 2)


class TestFrontierAndBasisPromotion(unittest.TestCase):
    def test_new_frontier_state_is_promoted_to_basis_after_two_rounds(self):
        tree = dfa_tree()
        # ('a',) is apart from the root (True vs False), so it has zero basis candidates once
        # check_frontier_consistency notices it; promotion of an isolated frontier state only
        # happens on the *next* round (promote_frontier_state runs before the dict is (re)populated).
        tree.insert_observation(['a'], [False])

        tree.update_frontier_and_basis()
        self.assertEqual(len(tree.basis), 1)
        self.assertEqual(len(tree.frontier_to_basis_dict), 1)
        sole_frontier_candidates = next(iter(tree.frontier_to_basis_dict.values()))
        self.assertEqual(sole_frontier_candidates, [])

        tree.update_frontier_and_basis()
        self.assertEqual(len(tree.basis), 2)
        self.assertEqual(tree.frontier_to_basis_dict, {})

    def test_find_basis_candidates_excludes_apart_states(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        node_a = tree.get_successor(['a'])
        # root.output is True, node_a.output is False: they are apart, so root cannot be a candidate.
        candidates = tree.find_basis_candidates(node_a)
        self.assertEqual(candidates, set())

    def test_make_basis_complete_explores_missing_alphabet_symbols(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.update_frontier_and_basis()
        tree.update_frontier_and_basis()
        self.assertEqual(len(tree.basis), 2)

        tree.make_basis_complete()
        for basis_state in tree.basis:
            for inp in tree.alphabet:
                self.assertIsNotNone(basis_state.get_successor(inp))


class TestConstructHypothesis(unittest.TestCase):
    def test_construct_hypothesis_reflects_current_tree_knowledge(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.update_frontier_and_basis()
        tree.update_frontier_and_basis()
        tree.make_basis_complete()
        tree.make_frontiers_identified()
        self.assertTrue(tree.is_observation_tree_adequate())

        hyp = tree.construct_hypothesis()
        self.assertEqual(len(hyp.states), 2)

        ground_truth = get_Angluin_dfa()
        for word in [tuple(), ('a',), ('b',), ('a', 'a'), ('a', 'b')]:
            hyp_out = hyp.execute_sequence(hyp.initial_state, list(word))
            gt_out = ground_truth.execute_sequence(ground_truth.initial_state, list(word))
            self.assertEqual(hyp_out, gt_out)


if __name__ == '__main__':
    unittest.main()
