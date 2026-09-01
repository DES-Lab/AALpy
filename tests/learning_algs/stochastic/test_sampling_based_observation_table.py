import unittest
from collections import defaultdict

from aalpy.learning_algs.stochastic.SamplingBasedObservationTable import SamplingBasedObservationTable


class _Teacher:
    initial_value = [False]

    def complete_query(self, s, e):
        return True


class _CompatibilityChecker:
    def __init__(self, different_pairs=frozenset()):
        self.different_pairs = different_pairs

    def are_cells_different(self, c1, c2, **kwargs):
        return (id(c1), id(c2)) in self.different_pairs or (id(c2), id(c1)) in self.different_pairs

    def use_diff_value(self):
        return False


def _reachable_states(automaton):
    reachable = set()
    stack = [automaton.initial_state]
    while stack:
        state = stack.pop()
        if state in reachable:
            continue
        reachable.add(state)
        for transitions in state.transitions.values():
            for target, probability in transitions:
                if probability and target not in reachable:
                    stack.append(target)
    return reachable


class StochasticObservationTableTest(unittest.TestCase):

    def test_mdp_representatives_are_access_closed(self):
        table = SamplingBasedObservationTable(
            ['a', 'b', 'c', 'd'], 'mdp', _Teacher(), _CompatibilityChecker()
        )

        initial = (False,)
        first = (False, 'd', False)
        second = (False, 'd', False, 'c', False)
        third = (False, 'd', False, 'c', False, 'b', False)
        discovered = (False, 'd', False, 'c', False, 'b', False, 'c', True)
        table.S = [initial, first, second, third, discovered]

        table.T = defaultdict(dict)
        for row in table.S:
            for inp in table.input_alphabet:
                table.T[row][inp] = {row[-1]: 10}

        table.T[initial][('d',)] = {False: 10}
        table.T[first][('c',)] = {False: 10}
        table.T[second][('b',)] = {False: 10}
        table.T[third][('c',)] = {False: 9, True: 1}

        table.update_compatibility_classes()
        self.assertTrue(set(table.S).issubset(table.compatibility_classes_representatives))

        hypothesis = table.generate_hypothesis()
        non_chaos_states = {state for state in hypothesis.states if state.output != 'chaos'}
        self.assertTrue(non_chaos_states.issubset(_reachable_states(hypothesis)))


class AreRowsCompatibleTest(unittest.TestCase):

    def test_mdp_rows_with_different_final_output_are_never_compatible(self):
        table = SamplingBasedObservationTable(['a'], 'mdp', _Teacher(), _CompatibilityChecker())
        s1 = (False,)
        s2 = (True,)
        table.T[s1][('a',)] = {False: 5}
        table.T[s2][('a',)] = {False: 5}
        # cells are identical (checker would say compatible), but final outputs differ
        self.assertFalse(table.are_rows_compatible(s1, s2))

    def test_smm_rows_use_checker_only(self):
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        s2 = ('a', 'x')
        table.T[s1][('a',)] = {'x': 5}
        table.T[s2][('a',)] = {'x': 5}
        self.assertTrue(table.are_rows_compatible(s1, s2))

    def test_rows_incompatible_when_checker_flags_a_cell(self):
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        s2 = ('a', 'x')
        cell1 = {'x': 5}
        cell2 = {'x': 1, 'y': 4}
        table.T[s1][('a',)] = cell1
        table.T[s2][('a',)] = cell2
        table.compatibility_checker = _CompatibilityChecker(different_pairs={(id(cell1), id(cell2))})
        self.assertFalse(table.are_rows_compatible(s1, s2))


class UpdateCompatibilityClassesTest(unittest.TestCase):

    def test_incompatible_rows_end_up_in_different_classes(self):
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        s2 = ('a', 'x')
        cell1 = {'x': 5}
        cell2 = {'x': 1, 'y': 4}
        table.S = [s1, s2]
        table.T[s1][('a',)] = cell1
        table.T[s2][('a',)] = cell2
        table.compatibility_checker = _CompatibilityChecker(different_pairs={(id(cell1), id(cell2))})

        table.update_compatibility_classes()

        self.assertEqual(set(table.compatibility_classes_representatives), {s1, s2})
        self.assertEqual(table.compatibility_class[s1], [])
        self.assertEqual(table.compatibility_class[s2], [])

    def test_compatible_rows_merge_into_a_single_class(self):
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        s2 = ('a', 'x')
        table.S = [s1, s2]
        table.T[s1][('a',)] = {'x': 5}
        table.T[s2][('a',)] = {'x': 3}

        table.update_compatibility_classes()

        self.assertEqual(len(table.compatibility_classes_representatives), 1)
        rep = table.compatibility_classes_representatives[0]
        other = s2 if rep == s1 else s1
        self.assertEqual(table.compatibility_class[rep], [other])


class GetRowToCloseTest(unittest.TestCase):

    def test_returns_none_when_all_extensions_are_covered(self):
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        table.S = [s1]
        # no output observed for input 'a' -> get_extended_s yields nothing
        table.T[s1][('a',)] = {}
        table.freq_query_cache[s1 + ('a',)] = {}
        table.update_compatibility_classes()
        self.assertIsNone(table.get_row_to_close())

    def test_returns_uncovered_extension_row(self):
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        lt = ('a', 'x')
        table.S = [s1]
        cell1 = {'x': 5}
        cell2 = {'y': 1}
        table.T[s1][('a',)] = cell1
        table.T[lt][('a',)] = cell2
        table.freq_query_cache[s1 + ('a',)] = cell1
        table.compatibility_checker = _CompatibilityChecker(different_pairs={(id(cell1), id(cell2))})
        table.update_compatibility_classes()
        self.assertEqual(table.get_row_to_close(), lt)


class GetConsistencyViolationTest(unittest.TestCase):

    def _build_inconsistent_table(self):
        # S = [s1, s2] compatible on E=[('a',)], but their extensions by (i='a', o='x') differ on the same
        # column, which is exactly a consistency violation.
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        s2 = ('a', 'x')
        table.S = [s1, s2]
        # rows are compatible: same content on the only column
        table.T[s1][('a',)] = {'x': 5}
        table.T[s2][('a',)] = {'x': 5}

        ext1 = s1 + ('a', 'x')  # ('a', 'x')
        ext2 = s2 + ('a', 'x')  # ('a', 'x', 'a', 'x')
        cell1 = {'y': 1}
        cell2 = {'z': 1}
        table.T[ext1][('a',)] = cell1
        table.T[ext2][('a',)] = cell2
        table.compatibility_checker = _CompatibilityChecker(different_pairs={(id(cell1), id(cell2))})
        return table, ext1, ext2

    def test_detects_violation(self):
        table, ext1, ext2 = self._build_inconsistent_table()
        violation = table.get_consistency_violation()
        self.assertEqual(violation, ('a', 'x', 'a'))

    def test_no_violation_when_extension_cells_agree(self):
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        s2 = ('a', 'x')
        table.S = [s1, s2]
        table.T[s1][('a',)] = {'x': 5}
        table.T[s2][('a',)] = {'x': 5}
        ext1 = s1 + ('a', 'x')
        ext2 = s2 + ('a', 'x')
        table.T[ext1][('a',)] = {'y': 1}
        table.T[ext2][('a',)] = {'y': 1}
        self.assertIsNone(table.get_consistency_violation())

    def test_none_when_cex_processing_enabled(self):
        table, _, _ = self._build_inconsistent_table()
        table.cex_processing = 'longest_prefix'
        self.assertIsNone(table.get_consistency_violation())


class GetUnambPercentageTest(unittest.TestCase):

    def test_representative_rows_are_unambiguous_but_unexplored_extensions_are_not(self):
        # S rows s1/s2 are in disjoint compatibility classes (unambiguous), but the freshly
        # discovered extension rows have no data in T yet, so they are (correctly) compatible
        # with every representative and therefore counted as ambiguous.
        table = SamplingBasedObservationTable(['a'], 'smm', _Teacher(), _CompatibilityChecker())
        s1 = ()
        s2 = ('a', 'x')
        cell1 = {'x': 5}
        cell2 = {'x': 1, 'y': 4}
        table.S = [s1, s2]
        table.T[s1][('a',)] = cell1
        table.T[s2][('a',)] = cell2
        table.freq_query_cache[s1 + ('a',)] = cell1
        table.freq_query_cache[s2 + ('a',)] = cell2
        table.compatibility_checker = _CompatibilityChecker(different_pairs={(id(cell1), id(cell2))})

        self.assertEqual(table.get_unamb_percentage(), 50.0)


if __name__ == '__main__':
    unittest.main()
