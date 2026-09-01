import unittest

from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import HoeffdingCompatibility
from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode


def _mc_node(freq: dict) -> AlergiaPtaNode:
    """Builds a standalone AlergiaPtaNode with plain-symbol (Alergia/MC-style) original frequencies."""
    node = AlergiaPtaNode(None, ())
    node.original_input_frequency = dict(freq)
    node.original_children = {k: AlergiaPtaNode(k, (k,)) for k in freq}
    return node


def _io_node(freq: dict) -> AlergiaPtaNode:
    """Builds a standalone AlergiaPtaNode with (input, output)-keyed (IOAlergia-style) original frequencies."""
    node = AlergiaPtaNode(None, ())
    node.original_input_frequency = dict(freq)
    node.original_children = {k: AlergiaPtaNode(k[1], (k,)) for k in freq}
    return node


class HoeffdingCompatibilityMcTest(unittest.TestCase):
    """For plain-symbol (Alergia) data, original_input_frequency keys are not tuples."""

    def test_identical_distributions_are_compatible(self):
        node_a = _mc_node({'x': 500, 'y': 500})
        node_b = _mc_node({'x': 500, 'y': 500})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertFalse(checker.are_states_different(node_a, node_b))

    def test_clearly_different_distributions_are_different(self):
        node_a = _mc_node({'x': 950, 'y': 50})
        node_b = _mc_node({'x': 50, 'y': 950})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertTrue(checker.are_states_different(node_a, node_b))

    def test_no_data_on_either_side_is_never_different(self):
        node_a = _mc_node({})
        node_b = _mc_node({'x': 10})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertFalse(checker.are_states_different(node_a, node_b))

    def test_smaller_eps_makes_the_bound_wider_and_less_sensitive(self):
        # A smaller eps means we demand more confidence before calling two distributions "different",
        # which widens the Hoeffding bound; the same borderline difference can therefore be flagged as
        # different for a larger eps but not for a much smaller one.
        node_a = _mc_node({'x': 5150, 'y': 4850})
        node_b = _mc_node({'x': 4850, 'y': 5150})
        larger_eps = HoeffdingCompatibility(eps=0.3)
        smaller_eps = HoeffdingCompatibility(eps=0.001)
        self.assertTrue(larger_eps.are_states_different(node_a, node_b))
        self.assertFalse(smaller_eps.are_states_different(node_a, node_b))


class HoeffdingCompatibilityIOAlergiaTest(unittest.TestCase):
    """For (input, output)-keyed (IOAlergia) data, the Hoeffding bound is checked per shared input."""

    def test_identical_conditional_output_distributions_are_compatible(self):
        node_a = _io_node({('a', 'x'): 500, ('a', 'y'): 500})
        node_b = _io_node({('a', 'x'): 500, ('a', 'y'): 500})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertFalse(checker.are_states_different(node_a, node_b))

    def test_different_conditional_output_distributions_are_different(self):
        node_a = _io_node({('a', 'x'): 950, ('a', 'y'): 50})
        node_b = _io_node({('a', 'x'): 50, ('a', 'y'): 950})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertTrue(checker.are_states_different(node_a, node_b))

    def test_disjoint_inputs_between_nodes_are_never_different(self):
        # get_immutable_inputs intersection is empty -> the per-input loop never runs -> not different,
        # regardless of how skewed each node's own distribution is
        node_a = _io_node({('a', 'x'): 950, ('a', 'y'): 50})
        node_b = _io_node({('b', 'x'): 50, ('b', 'y'): 950})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertFalse(checker.are_states_different(node_a, node_b))

    def test_different_output_for_shared_input_is_detected_even_with_extra_disjoint_input(self):
        node_a = _io_node({('a', 'x'): 950, ('a', 'y'): 50, ('c', 'z'): 10})
        node_b = _io_node({('a', 'x'): 50, ('a', 'y'): 950})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertTrue(checker.are_states_different(node_a, node_b))

    def test_no_data_on_either_side_is_never_different(self):
        node_a = _io_node({})
        node_b = _io_node({('a', 'x'): 10})
        checker = HoeffdingCompatibility(eps=0.05)
        self.assertFalse(checker.are_states_different(node_a, node_b))


if __name__ == '__main__':
    unittest.main()
