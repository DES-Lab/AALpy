import unittest

from aalpy.learning_algs.general_passive.DataHandler import CountDataHandler, CountOnPTADataHandler


def counting_pta(handler, data_format, data):
    return handler.createPTA(data, 'moore', data_format=data_format)


class TestCountDataHandler(unittest.TestCase):
    def test_copy_can_be_merged_into(self):
        dh = CountDataHandler()
        x, y = dh.init_data(), dh.init_data()
        dh.aggregate_data(_node_with(x), 'a', 'x', None)
        dh.aggregate_data(_node_with(y), 'b', 'y', None)
        # merging into a copy must work for inputs the copy has never seen
        merged = dh.merge(dh.copy(x), y)
        self.assertEqual(dict(merged.transition_count), {'a': {'x': 1}, 'b': {'y': 1}})

    def test_copy_is_independent_of_original(self):
        dh = CountDataHandler()
        x = dh.init_data()
        dh.aggregate_data(_node_with(x), 'a', 'x', None)
        copy = dh.copy(x)
        dh.aggregate_data(_node_with(x), 'a', 'x', None)
        self.assertEqual(copy.transition_count['a'], {'x': 1})


class TestCountOnPTADataHandler(unittest.TestCase):
    def test_copy_keeps_pta_data(self):
        dh = CountOnPTADataHandler()
        pta = counting_pta(dh, 'labeled_sequences', [(('a',), True), (('b',), False)])
        copy = dh.copy(pta.data)
        self.assertIsInstance(copy, type(pta.data))
        self.assertEqual(copy.pta_count, pta.data.pta_count)
        self.assertEqual(copy.shadow_pta, pta.data.shadow_pta)


class TestCreatePTA(unittest.TestCase):
    def test_labeled_sequences_initialize_node_data(self):
        dh = CountOnPTADataHandler()
        pta = counting_pta(dh, 'labeled_sequences', [(('a', 'b'), True), (('a',), False)])
        for node in pta.get_all_nodes():
            self.assertIsNotNone(node.data)

    @unittest.expectedFailure
    def test_labeled_sequences_and_io_traces_count_the_same_transitions(self):
        # known gap: add_labeled_sequence aggregates intermediate steps under unknown_output and only
        # remaps `transitions` when the output is resolved later, so the counts stay split
        labeled = [(('a', 'b'), True), (('a',), False)]
        io_traces = [[False, ('a', False), ('b', True)], [False, ('a', False)]]
        pta_labeled = counting_pta(CountDataHandler(), 'labeled_sequences', labeled)
        pta_traces = counting_pta(CountDataHandler(), 'io_traces', io_traces)
        self.assertEqual(dict(pta_labeled.data.transition_count), dict(pta_traces.data.transition_count))


def _node_with(data):
    """Minimal stand-in for the source node of aggregate_data, which only accesses .data."""

    class Node:
        pass

    node = Node()
    node.data = data
    return node


if __name__ == '__main__':
    unittest.main()
