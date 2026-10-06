import unittest

from aalpy.learning_algs.general_passive.DataHandler import CountDataHandler, CountOnPTADataHandler
from aalpy.learning_algs.general_passive.GsmNode import GsmNode


def counting_pta(handler, data_format, data):
    return handler.createPTA(data, 'moore', data_format=data_format)


class TestCountDataHandler(unittest.TestCase):
    def test_copy_can_be_merged_into(self):
        dh = CountDataHandler()
        x, y = dh.init_data(), dh.init_data()
        dh.aggregate_data(_node_with(x), 'a', 'x', GsmNode(('a', 'x'), None, None))
        dh.aggregate_data(_node_with(y), 'b', 'y', GsmNode(('b', 'y'), None, None))
        # merging into a copy must work for inputs the copy has never seen
        merged = dh.merge(dh.copy(x), y)
        self.assertEqual(dict(merged.transition_count), {'a': {'x': 1}, 'b': {'y': 1}})

    def test_copy_is_independent_of_original(self):
        dh = CountDataHandler()
        x = dh.init_data()
        dh.aggregate_data(_node_with(x), 'a', 'x', GsmNode(('a', 'x'), None, None))
        copy = dh.copy(x)
        dh.aggregate_data(_node_with(x), 'a', 'x', GsmNode(('a', 'x'), None, None))
        self.assertEqual(copy.transition_count['a'], {'x': 1})

    def test_counts_use_abstracted_symbols(self):
        for base in (CountDataHandler, CountOnPTADataHandler):
            class Rounding(base):
                def abstract(self, in_val, out_val):
                    return in_val, round(out_val)

            root = Rounding().createPTA([[('a', 1.2)], [('a', 0.9)]], 'mealy', 'io_traces')
            self.assertEqual(dict(root.data.transition_count), {'a': {1: 2}})
            smm = root.to_automaton('mealy', 'stochastic')
            self.assertEqual([(out, prob) for _, out, prob in smm.initial_state.transitions['a']], [(1, 1.0)])


class TestCountOnPTADataHandler(unittest.TestCase):
    def test_copy_keeps_pta_data(self):
        dh = CountOnPTADataHandler()
        pta = counting_pta(dh, 'io_traces', [[True, ('a',True), ('b', False)]])
        copy = dh.copy(pta.data)
        self.assertIsInstance(copy, type(pta.data))
        self.assertEqual(copy.pta_count, pta.data.pta_count)
        self.assertEqual(copy.shadow_pta, pta.data.shadow_pta)


class TestCreatePTA(unittest.TestCase):
    def test_labeled_sequences_initialize_node_data(self):
        dh = CountOnPTADataHandler()
        pta = counting_pta(dh, 'io_traces', [[True, ('a',True), ('b', False)]])
        for node in pta.get_all_nodes():
            self.assertIsNotNone(node.data)

    def test_conflicting_empty_input_labels_raise(self):
        with self.assertRaises(ValueError):
            counting_pta(CountDataHandler(), 'labeled_sequences', [([], 'a'), ([], 'b')])

def _node_with(data):
    """Minimal stand-in for the source node of aggregate_data, which only accesses .data."""

    class Node:
        pass

    node = Node()
    node.data = data
    return node


if __name__ == '__main__':
    unittest.main()
