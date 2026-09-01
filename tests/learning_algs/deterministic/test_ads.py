import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs.deterministic.ADS import Ads
from aalpy.learning_algs.deterministic.ObservationTree import ObservationTree
from aalpy.utils import get_Angluin_dfa


def dfa_tree():
    sul = AutomatonSUL(get_Angluin_dfa())
    return ObservationTree(['a', 'b'], sul, 'dfa', 'ADS', 'ADS')


def mealy_tree():
    sul = AutomatonSUL(get_Angluin_dfa())
    return ObservationTree(['a', 'b'], sul, 'mealy', 'ADS', 'ADS')


class TestAdsSingleNodeBlock(unittest.TestCase):
    def test_single_node_block_is_a_leaf_with_zero_score(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [True])
        node_a = tree.get_successor(['a'])
        ads = Ads(tree, [node_a])
        self.assertEqual(ads.get_score(), 0)
        self.assertIsNone(ads.next_input(None))


class TestAdsMooreDfa(unittest.TestCase):
    def test_immediate_own_output_difference_gets_maximal_score(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [True])
        tree.insert_observation(['b'], [False])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])

        ads = Ads(tree, [node_a, node_b])
        self.assertEqual(ads.get_score(), 1.0)
        # The block splits purely on the states' own (already-known) output; the root of the ADS
        # uses the tuple() sentinel to represent this (no real input is sent).
        self.assertEqual(ads.next_input(None), tuple())

    def test_ads_correctly_separates_states_needing_one_step_lookahead(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.insert_observation(['b'], [False])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        tree.insert_observation(['a', 'a'], [False, True])
        tree.insert_observation(['b', 'a'], [False, False])

        ads = Ads(tree, [node_a, node_b])
        self.assertEqual(ads.get_score(), 1.0)

        for node, expected_final_output in [(node_a, True), (node_b, False)]:
            ads.reset_to_root()
            first_input = ads.next_input(None)
            self.assertEqual(first_input, tuple())
            second_input = ads.next_input(node.output)
            self.assertEqual(second_input, 'a')
            successor_output = node.get_successor(second_input).output
            self.assertEqual(successor_output, expected_final_output)
            self.assertIsNone(ads.next_input(successor_output))

    def test_no_information_available_falls_back_to_a_zero_score_guess(self):
        """
        construct_ads_rec's "no successor anywhere in the block" guard (see its docstring/comment)
        is meant to raise when it truly cannot pick a next input. In practice it never fires: it
        checks `node.successors is not None`, but successors is always a dict (defaulting to {},
        never None), so the condition is always true regardless of whether any real successor was
        recorded. Confirmed here: when node_a/node_b share the same own output and neither has any
        recorded successor, construction does not raise -- it falls back to picking the first
        alphabet symbol with score 0 and an empty (non-distinguishing) child map. This fallback
        actually matters in practice (e.g. for run_Lsharp(..., 'moore', extension_rule='ADS',
        separation_rule='ADS')): it lets the algorithm still send a real input to the SUL to gather
        more information instead of hard failing when the tree does not yet have enough data, so
        this is intentionally left as-is rather than "fixed".
        """
        tree = dfa_tree()
        tree.insert_observation(['a'], [True])
        tree.insert_observation(['b'], [True])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])

        ads = Ads(tree, [node_a, node_b])
        self.assertEqual(ads.get_score(), 0.0)
        # First step just splits on the (identical) own output via the tuple() sentinel; the actual
        # zero-score fallback input is one level down, in the subtree for that shared output.
        self.assertEqual(ads.next_input(None), tuple())
        self.assertEqual(ads.next_input(True), tree.alphabet[0])


class TestAdsMealy(unittest.TestCase):
    def test_single_input_perfectly_separates_three_state_block(self):
        tree = mealy_tree()
        tree.insert_observation(['a'], [1])
        tree.insert_observation(['b'], [2])
        tree.insert_observation(['a', 'b'], [1, 9])
        node_1 = tree.get_successor(['a'])
        node_2 = tree.get_successor(['b'])
        node_3 = tree.get_successor(['a', 'b'])

        tree.insert_observation(['a', 'a'], [1, 5])
        tree.insert_observation(['b', 'a'], [2, 6])
        tree.insert_observation(['a', 'b', 'a'], [1, 9, 7])

        ads = Ads(tree, [node_1, node_2, node_3])
        # 3 states perfectly separated by one input reaches the maximal possible score, len(block)-1.
        self.assertEqual(ads.get_score(), 2.0)

        outputs_seen = set()
        for node in [node_1, node_2, node_3]:
            ads.reset_to_root()
            first_input = ads.next_input(None)
            self.assertEqual(first_input, 'a')
            output = node.get_output(first_input)
            outputs_seen.add(output)
            self.assertIsNone(ads.next_input(output))
        # each state produces a distinct output on the chosen input, so the ADS fully identifies them
        self.assertEqual(len(outputs_seen), 3)

    def test_maximal_base_input_prefers_the_more_discriminating_input(self):
        tree = mealy_tree()
        tree.insert_observation(['a'], [1])
        tree.insert_observation(['b'], [2])
        node_1 = tree.get_successor(['a'])
        node_2 = tree.get_successor(['b'])
        # 'a' does not distinguish the two nodes (same output on 'a'), 'b' does.
        tree.insert_observation(['a', 'a'], [1, 0])
        tree.insert_observation(['b', 'a'], [2, 0])
        tree.insert_observation(['a', 'b'], [1, 3])
        tree.insert_observation(['b', 'b'], [2, 4])

        ads = Ads(tree, [node_1, node_2])
        best_input, best_score = ads.maximal_base_input(['a', 'b'], [node_1, node_2], 'mealy')
        self.assertEqual(best_input, 'b')
        self.assertEqual(best_score, 1.0)


if __name__ == '__main__':
    unittest.main()
