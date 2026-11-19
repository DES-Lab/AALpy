import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs.deterministic.ClassificationTree import ClassificationTree, CTInternalNode, CTLeafNode
from aalpy.utils import get_Angluin_dfa
from aalpy.utils.ModelChecking import bisimilar


def find_counterexample(sul, hypothesis, max_length=6):
    import itertools
    alphabet = ['a', 'b']
    for length in range(1, max_length + 1):
        for word in itertools.product(alphabet, repeat=length):
            hypothesis.reset_to_initial()
            hyp_out = hypothesis.execute_sequence(hypothesis.initial_state, list(word))
            sul_out = sul.query(word)
            if hyp_out[-1] != sul_out[-1]:
                return tuple(word)
    raise AssertionError('no counterexample found within max_length')


class TestClassificationTreeInitDfa(unittest.TestCase):
    def test_root_has_two_leaves_for_initial_state_and_cex(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        self.assertIsInstance(ct.root, CTInternalNode)
        self.assertEqual(ct.root.distinguishing_string, tuple())
        self.assertIn(tuple(), ct.leaf_nodes)
        self.assertIn(('a',), ct.leaf_nodes)
        self.assertEqual(len(ct.root.children), 2)

    def test_initial_hypothesis_has_two_states(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        hyp = ct.update_hypothesis()
        self.assertEqual(len(hyp.states), 2)
        prefixes = {s.prefix for s in hyp.states}
        self.assertEqual(prefixes, {tuple(), ('a',)})


class TestClassificationTreeInitMealy(unittest.TestCase):
    def test_root_distinguishing_string_is_last_symbol_of_cex(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'mealy', cex=('a', 'a'))
        self.assertEqual(ct.root.distinguishing_string, ('a',))

    def test_initial_hypothesis_output_fun_matches_sul(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'mealy', cex=('a', 'a'))
        hyp = ct.update_hypothesis()
        self.assertEqual(len(hyp.states), 2)
        initial = next(s for s in hyp.states if s.prefix == tuple())
        self.assertEqual(initial.output_fun['a'], sul.query(('a',))[-1])
        self.assertEqual(initial.output_fun['b'], sul.query(('b',))[-1])


class TestSift(unittest.TestCase):
    def test_sift_routes_known_access_strings_to_themselves(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        ct.update_hypothesis()
        self.assertEqual(ct._sift(tuple()).access_string, tuple())
        self.assertEqual(ct._sift(('a',)).access_string, ('a',))

    def test_sift_routes_equivalent_word_to_existing_leaf(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        ct.update_hypothesis()
        # 'b' (-> q2, non-accepting) is indistinguishable from ('a',) (-> q1, non-accepting) under
        # the root's current distinguishing string (the empty word), so it sifts to the same leaf.
        self.assertEqual(ct._sift(('b',)).access_string, ('a',))

    def test_sift_of_equivalent_word_does_not_grow_the_tree(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        ct.update_hypothesis()
        leaves_before = set(ct.leaf_nodes)
        # ('b', 'b') reaches q0 (accepting) again via a different path than any known access string,
        # but since the tree currently only distinguishes accepting/non-accepting, sifting it does
        # not introduce a new leaf -- it just routes to the existing empty-word-labelled leaf.
        leaf = ct._sift(('b', 'b'))
        self.assertIsInstance(leaf, CTLeafNode)
        self.assertEqual(leaf.access_string, tuple())
        self.assertEqual(set(ct.leaf_nodes), leaves_before)

    def test_sift_creates_a_new_leaf_the_first_time_a_branch_is_taken(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        ct.update_hypothesis()

        # Replace the tree with a hand-built root discriminated by suffix ('a',) that only has the
        # True branch populated, to directly control and observe _sift's leaf-creation branch.
        new_root = CTInternalNode(distinguishing_string=('a',), parent=None, path_to_node=None)
        leaf_true = CTLeafNode(access_string=tuple(), parent=new_root, path_to_node=True)
        new_root.children[True] = leaf_true
        ct.root = new_root
        ct.leaf_nodes = {tuple(): leaf_true}

        # query(('b',) + ('a',)) = query(('b', 'a')): q0 -b-> q2 -a-> q3, output False -- a key not
        # yet present among new_root's children, so sifting 'b' must create and register a new leaf.
        leaf = ct._sift(('b',))
        self.assertEqual(leaf.access_string, ('b',))
        self.assertEqual(leaf.path_to_node, False)
        self.assertIn(('b',), ct.leaf_nodes)
        self.assertIs(new_root.children[False], leaf)


class TestProcessCounterexampleRs(unittest.TestCase):
    def test_process_counterexample_grows_the_tree_and_hypothesis(self):
        dfa = get_Angluin_dfa()
        sul = AutomatonSUL(dfa)
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        hyp = ct.update_hypothesis()
        self.assertEqual(len(hyp.states), 2)

        cex = find_counterexample(sul, hyp)
        ct.process_counterexample(cex, hyp, 'rs')
        hyp = ct.update_hypothesis()
        self.assertEqual(len(hyp.states), 3)

    def test_learning_converges_to_ground_truth_after_enough_counterexamples(self):
        dfa = get_Angluin_dfa()
        sul = AutomatonSUL(dfa)
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        hyp = ct.update_hypothesis()

        for _ in range(10):
            try:
                cex = find_counterexample(sul, hyp)
            except AssertionError:
                break
            ct.process_counterexample(cex, hyp, 'rs')
            hyp = ct.update_hypothesis()

        self.assertEqual(len(hyp.states), len(dfa.states))
        self.assertTrue(bisimilar(dfa, hyp))


class TestLegacyUpdateMethod(unittest.TestCase):
    """
    ClassificationTree.update() (distinct from process_counterexample()) implements a second,
    unparametrized counterexample-processing strategy. It is not called anywhere in KV.py (which
    always uses process_counterexample), but it is still part of the class's public surface, so it
    is exercised directly here.
    """

    def test_update_grows_the_tree_and_hypothesis(self):
        dfa = get_Angluin_dfa()
        sul = AutomatonSUL(dfa)
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        hyp = ct.update_hypothesis()
        self.assertEqual(len(hyp.states), 2)

        cex = find_counterexample(sul, hyp)
        ct.update(cex, hyp)
        hyp = ct.update_hypothesis()
        self.assertEqual(len(hyp.states), 3)


class TestLeastCommonAncestor(unittest.TestCase):
    def test_lca_of_siblings_is_the_root(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        ct = ClassificationTree(['a', 'b'], sul, 'dfa', cex=('a',))
        ct.update_hypothesis()
        self.assertEqual(ct._least_common_ancestor(tuple(), ('a',)), ct.root.distinguishing_string)


if __name__ == '__main__':
    unittest.main()
