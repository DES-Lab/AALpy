import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs.non_deterministic.AbstractedOnfsmObservationTable import AbstractedNonDetObservationTable
from aalpy.learning_algs.non_deterministic.NonDeterministicSULWrapper import NonDeterministicSULWrapper
from aalpy.utils import get_benchmark_ONFSM

ABSTRACTION = {0: 'even', 2: 'even', 3: 'odd'}


def wrapped_sul():
    return NonDeterministicSULWrapper(AutomatonSUL(get_benchmark_ONFSM()))


def initial_round(at):
    """Mirrors the first few lines of run_abstracted_ONFSM_Lstar: the initial row must be queried
    and abstracted before S_dot_A/T are populated enough for get_row_to_close() to work."""
    at.update_obs_table()
    new_rows = at.update_extended_S()
    at.update_obs_table(s_set=new_rows)


class TestConstruction(unittest.TestCase):
    def test_initializes_empty_S_dot_A_and_E(self):
        at = AbstractedNonDetObservationTable(['a', 'b'], wrapped_sul(), ABSTRACTION, 10)
        self.assertEqual(at.S, [((), ())])
        self.assertEqual(at.S_dot_A, [])
        self.assertEqual(at.E, [])
        self.assertEqual(at.A, [('a',), ('b',)])

    def test_wraps_a_plain_non_det_observation_table_internally(self):
        at = AbstractedNonDetObservationTable(['a', 'b'], wrapped_sul(), ABSTRACTION, 10)
        self.assertEqual(at.observation_table.alphabet, ['a', 'b'])

    def test_asserts_on_missing_alphabet_or_sul(self):
        with self.assertRaises(AssertionError):
            AbstractedNonDetObservationTable(None, wrapped_sul(), ABSTRACTION, 10)
        with self.assertRaises(AssertionError):
            AbstractedNonDetObservationTable(['a'], None, ABSTRACTION, 10)


class TestGetAbstraction(unittest.TestCase):
    def test_maps_known_outputs_to_their_equivalence_class(self):
        at = AbstractedNonDetObservationTable(['a', 'b'], wrapped_sul(), ABSTRACTION, 10)
        self.assertEqual(at.get_abstraction(0), 'even')
        self.assertEqual(at.get_abstraction(2), 'even')
        self.assertEqual(at.get_abstraction(3), 'odd')

    def test_falls_back_to_the_original_output_when_unmapped(self):
        at = AbstractedNonDetObservationTable(['a', 'b'], wrapped_sul(), ABSTRACTION, 10)
        self.assertEqual(at.get_abstraction('unmapped_output'), 'unmapped_output')


class TestAbstractObsTable(unittest.TestCase):
    def test_T_contains_abstracted_outputs_not_raw_ones(self):
        random.seed(0)
        sul = wrapped_sul()
        at = AbstractedNonDetObservationTable(sul.sul.automaton.get_input_alphabet(), sul, ABSTRACTION, 20)
        at.update_obs_table()

        raw_traces = sul.cache.get_all_traces(((), ()), ('a',))
        self.assertTrue(any(t[0] in (0, 2, 3) for t in raw_traces))

        abstracted_cell = at.T[((), ())][('a',)]
        for value in abstracted_cell:
            self.assertIn(value, [('even',), ('odd',)])

    def test_raw_cache_keeps_concrete_outputs_alongside_the_abstraction(self):
        random.seed(0)
        sul = wrapped_sul()
        alphabet = sul.sul.automaton.get_input_alphabet()
        at = AbstractedNonDetObservationTable(alphabet, sul, ABSTRACTION, 20)
        at.update_obs_table()
        raw_cell = sul.cache.get_all_traces(((), ()), ('a',))
        self.assertIn((0,), raw_cell)


class TestGetRowToClose(unittest.TestCase):
    def test_moves_a_row_from_S_dot_A_into_S(self):
        random.seed(0)
        sul = wrapped_sul()
        alphabet = sul.sul.automaton.get_input_alphabet()
        at = AbstractedNonDetObservationTable(alphabet, sul, ABSTRACTION, 20)
        initial_round(at)

        s_before = list(at.S)
        s_dot_a_before = list(at.S_dot_A)
        row = at.get_row_to_close()

        self.assertIsNotNone(row)
        self.assertIn(row, at.S)
        self.assertNotIn(row, at.S_dot_A)
        self.assertIn(row, s_dot_a_before)
        self.assertNotIn(row, s_before)

    def test_repeated_calls_do_not_require_re_querying_in_between(self):
        # Unlike NonDetObservationTable.get_row_to_close, which needs a fresh
        # query_missing_observations() call between invocations, this abstracted variant works off
        # S_dot_A directly, so consecutive calls can each pull a different row without re-querying.
        random.seed(0)
        sul = wrapped_sul()
        alphabet = sul.sul.automaton.get_input_alphabet()
        at = AbstractedNonDetObservationTable(alphabet, sul, ABSTRACTION, 20)
        initial_round(at)

        first = at.get_row_to_close()
        second = at.get_row_to_close()
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)


class TestCleanObsTable(unittest.TestCase):
    def test_removes_duplicate_rows_from_S(self):
        random.seed(0)
        sul = wrapped_sul()
        alphabet = sul.sul.automaton.get_input_alphabet()
        at = AbstractedNonDetObservationTable(alphabet, sul, ABSTRACTION, 20)
        initial_round(at)

        row_to_close = at.get_row_to_close()
        while row_to_close is not None:
            row_to_close = at.get_row_to_close()

        sizes_before = len(at.S)
        at.clean_obs_table()
        self.assertLessEqual(len(at.S), sizes_before)


class TestExtendSDotA(unittest.TestCase):
    def test_only_adds_prefixes_not_already_present(self):
        at = AbstractedNonDetObservationTable(['a'], wrapped_sul(), ABSTRACTION, 10)
        cex_prefixes = [(('a',), (0,))]
        added_first = at.extend_S_dot_A(cex_prefixes)
        added_second = at.extend_S_dot_A(cex_prefixes)
        self.assertEqual(added_first, cex_prefixes)
        self.assertEqual(added_second, [])
        self.assertEqual(at.S_dot_A.count((('a',), (0,))), 1)

    def test_does_not_add_prefixes_already_in_S(self):
        at = AbstractedNonDetObservationTable(['a'], wrapped_sul(), ABSTRACTION, 10)
        added = at.extend_S_dot_A([((), ())])
        self.assertEqual(added, [])
        self.assertEqual(at.S_dot_A, [])


if __name__ == '__main__':
    unittest.main()
