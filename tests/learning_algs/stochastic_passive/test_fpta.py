import unittest

from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode, create_fpta


class CreateFptaMcTest(unittest.TestCase):

    def test_single_sequence_builds_a_chain(self):
        # for 'mc', like 'mdp', the first element of each sequence is treated as the (shared) initial output
        # of the root and is not itself a transition
        data = [['a', 'b', 'a']]
        root = create_fpta(data, 'mc')

        self.assertEqual(root.prefix, ())
        self.assertEqual(root.output, 'a')
        self.assertEqual(root.input_frequency, {'b': 1})

        node_b = root.children['b']
        self.assertEqual(node_b.output, 'b')
        self.assertEqual(node_b.prefix, ('b',))
        self.assertEqual(node_b.input_frequency, {'a': 1})

        node_ba = node_b.children['a']
        self.assertEqual(node_ba.output, 'a')
        self.assertEqual(node_ba.children, {})

    def test_shared_prefixes_merge_into_a_tree(self):
        data = [['a', 'b'], ['a', 'c'], ['a', 'b']]
        root = create_fpta(data, 'mc')

        self.assertEqual(root.output, 'a')
        self.assertEqual(root.input_frequency, {'b': 2, 'c': 1})
        self.assertEqual(set(root.children.keys()), {'b', 'c'})


class CreateFptaMdpTest(unittest.TestCase):

    def test_initial_output_is_first_element_of_first_sequence(self):
        data = [['A', ('a', 'B'), ('b', 'A')], ['A', ('a', 'C')]]
        root = create_fpta(data, 'mdp')

        self.assertEqual(root.output, 'A')
        self.assertEqual(root.prefix, ())
        self.assertEqual(root.input_frequency, {('a', 'B'): 1, ('a', 'C'): 1})

        node_b = root.children[('a', 'B')]
        self.assertEqual(node_b.output, 'B')
        self.assertEqual(node_b.prefix, (('a', 'B'),))
        self.assertEqual(node_b.input_frequency, {('b', 'A'): 1})

        node_c = root.children[('a', 'C')]
        self.assertEqual(node_c.output, 'C')
        self.assertEqual(node_c.children, {})

    def test_inconsistent_initial_output_raises(self):
        data = [['A', ('a', 'B')], ['X', ('a', 'B')]]
        with self.assertRaises(AssertionError):
            create_fpta(data, 'mdp')


class CreateFptaSmmTest(unittest.TestCase):

    def test_no_initial_output_and_outputs_are_none(self):
        data = [['a', 'o1', 'b', 'o2'], ['a', 'o1', 'a', 'o3']]
        root = create_fpta(data, 'smm')

        self.assertIsNone(root.output)
        self.assertEqual(root.prefix, ())
        self.assertEqual(root.input_frequency, {'a': 2})

        node = root.children['a']
        # for smm the output is never inferred from the input element itself
        self.assertIsNone(node.output)
        self.assertEqual(node.input_frequency, {'o1': 2})

        node2 = node.children['o1']
        self.assertEqual(set(node2.children.keys()), {'b', 'a'})


class AlergiaPtaNodeHelpersTest(unittest.TestCase):

    def test_get_input_frequency_sums_over_outputs(self):
        node = AlergiaPtaNode(None, ())
        node.input_frequency = {('a', 'x'): 3, ('a', 'y'): 2, ('b', 'x'): 5}
        self.assertEqual(node.get_input_frequency('a'), 5)
        self.assertEqual(node.get_input_frequency('b'), 5)
        self.assertEqual(node.get_input_frequency('c'), 0)

    def test_get_output_frequencies_filters_by_input(self):
        node = AlergiaPtaNode(None, ())
        node.input_frequency = {('a', 'x'): 3, ('a', 'y'): 2, ('b', 'x'): 5}
        self.assertEqual(node.get_output_frequencies('a'), {'x': 3, 'y': 2})

    def test_get_inputs_uses_mutable_input_frequency(self):
        node = AlergiaPtaNode(None, ())
        node.input_frequency = {('a', 'x'): 1, ('b', 'y'): 1}
        self.assertEqual(node.get_inputs(), {'a', 'b'})

    def test_successors_returns_children_values(self):
        # AlergiaPtaNode defines __eq__ by prefix but is unhashable, so successors are compared by identity
        parent = AlergiaPtaNode(None, ())
        child1 = AlergiaPtaNode('x', ('a',))
        child2 = AlergiaPtaNode('y', ('b',))
        parent.children = {'a': child1, 'b': child2}
        successors = parent.successors()
        self.assertEqual(len(successors), 2)
        self.assertTrue(any(s is child1 for s in successors))
        self.assertTrue(any(s is child2 for s in successors))

    def test_ordering_is_by_prefix_length_then_value(self):
        short = AlergiaPtaNode(None, ('a',))
        long = AlergiaPtaNode(None, ('a', 'b'))
        other_short = AlergiaPtaNode(None, ('z',))
        self.assertLess(short, long)
        self.assertLess(short, other_short)
        self.assertLessEqual(short, short)

    def test_equality_is_based_on_prefix_only(self):
        node1 = AlergiaPtaNode('x', ('a', 'b'))
        node2 = AlergiaPtaNode('y', ('a', 'b'))
        self.assertEqual(node1, node2)


if __name__ == '__main__':
    unittest.main()
