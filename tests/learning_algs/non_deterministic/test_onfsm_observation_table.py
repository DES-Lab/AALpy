import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Onfsm, OnfsmState
from aalpy.learning_algs.non_deterministic.NonDeterministicSULWrapper import NonDeterministicSULWrapper
from aalpy.learning_algs.non_deterministic.OnfsmObservationTable import NonDetObservationTable


def branching_onfsm():
    """s0 --a/x--> s1 --a/x--> s0 and s0 --a/y--> s2 --a/y--> s0.
    A single-input-letter ONFSM whose two non-initial states are distinguishable by the suffix
    'a' (s1 always answers 'x', s2 always answers 'y')."""
    s0 = OnfsmState('s0')
    s1 = OnfsmState('s1')
    s2 = OnfsmState('s2')
    s0.transitions['a'].append(('x', s1))
    s0.transitions['a'].append(('y', s2))
    s1.transitions['a'].append(('x', s0))
    s2.transitions['a'].append(('y', s0))
    return Onfsm(s0, [s0, s1, s2])


def wrapped_sul(onfsm):
    return NonDeterministicSULWrapper(AutomatonSUL(onfsm))


def close_table(ot):
    """Drives the table-closing loop the same way run_non_det_Lstar does: querying missing
    observations must be interleaved with get_row_to_close(), since get_extended_S() (and hence
    get_row_to_close) only reflects rows the cache has actually been queried for so far."""
    ot.query_missing_observations()
    row_to_close = ot.get_row_to_close()
    while row_to_close is not None:
        ot.query_missing_observations()
        row_to_close = ot.get_row_to_close()
        ot.clean_obs_table()


class TestConstruction(unittest.TestCase):
    def test_initializes_A_and_E_from_alphabet(self):
        ot = NonDetObservationTable(['a', 'b'], wrapped_sul(branching_onfsm()), 5)
        self.assertEqual(ot.A, [('a',), ('b',)])
        self.assertEqual(ot.E, [('a',), ('b',)])

    def test_S_starts_with_only_the_empty_row(self):
        ot = NonDetObservationTable(['a'], wrapped_sul(branching_onfsm()), 5)
        self.assertEqual(ot.S, [((), ())])

    def test_asserts_on_missing_alphabet_or_sul(self):
        with self.assertRaises(AssertionError):
            NonDetObservationTable(None, wrapped_sul(branching_onfsm()), 5)
        with self.assertRaises(AssertionError):
            NonDetObservationTable(['a'], None, 5)


class TestQueryMissingObservations(unittest.TestCase):
    def test_samples_initial_row_at_least_n_times_per_column(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        ot.query_missing_observations()
        freq = sul.cache.get_s_e_sampling_frequency(((), ()), ('a',))
        self.assertGreaterEqual(freq, 15)

    def test_first_call_does_not_reach_beyond_depth_one(self):
        # get_extended_S() is computed from the cache as it stands *before* query_missing_observations
        # runs, so on a fresh table the very first call only samples the empty-prefix row - deeper
        # rows only get discovered (and then sampled) on subsequent calls.
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        ot.query_missing_observations()
        self.assertEqual(sul.cache.get_all_traces((('a',), ('x',)), ('a',)), [])


class TestGetRowToClose(unittest.TestCase):
    def test_returns_none_for_a_freshly_closed_single_state_row(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        ot.query_missing_observations()
        row = ot.get_row_to_close()
        self.assertIsNotNone(row)
        self.assertIn(row, ot.S)

    def test_full_closing_loop_reaches_a_fixed_point(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)
        self.assertIsNone(ot.get_row_to_close())

    def test_closing_loop_finds_three_distinguishable_rows(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)
        self.assertEqual(len(ot.S), 3)
        hashes = {ot.row_to_hashable(s) for s in ot.S}
        self.assertEqual(len(hashes), 3)


class TestRowToHashable(unittest.TestCase):
    def test_distinguishes_rows_with_different_observed_outputs(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)
        s1_row = next(s for s in ot.S if s[1] == ('x',))
        s2_row = next(s for s in ot.S if s[1] == ('y',))
        self.assertEqual(ot.row_to_hashable(s1_row), (frozenset({('x',)}),))
        self.assertEqual(ot.row_to_hashable(s2_row), (frozenset({('y',)}),))

    def test_initial_row_reflects_both_branches(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)
        self.assertEqual(ot.row_to_hashable(((), ())), (frozenset({('x',), ('y',)}),))


class TestCleanObsTable(unittest.TestCase):
    def test_removes_duplicate_rows_that_loop_back_to_an_existing_state(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)

        # ('a','a') with outputs ('y','y') loops back to s0 and is thus a duplicate of the empty row
        duplicate_row = (('a', 'a'), ('y', 'y'))
        ot.S.append(duplicate_row)
        ot.query_missing_observations([duplicate_row], ot.E)

        ot.clean_obs_table()
        self.assertNotIn(duplicate_row, ot.S)
        self.assertEqual(len(ot.S), 3)


class TestGenHypothesis(unittest.TestCase):
    def test_learned_table_yields_an_equivalent_three_state_onfsm(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)

        hypothesis = ot.gen_hypothesis()
        self.assertIsInstance(hypothesis, Onfsm)
        self.assertEqual(len(hypothesis.states), 3)
        self.assertCountEqual(hypothesis.outputs_on_input('a'), ['x', 'y'])

    def test_hypothesis_round_trips_a_known_trace(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)
        hypothesis = ot.gen_hypothesis()

        hypothesis.reset_to_initial()
        self.assertEqual(hypothesis.step_to('a', 'x'), 'x')
        self.assertEqual(hypothesis.step_to('a', 'x'), 'x')
        self.assertIsNone(hypothesis.step_to('a', 'z'))

    def test_hypothesis_carries_characterization_set(self):
        random.seed(0)
        sul = wrapped_sul(branching_onfsm())
        ot = NonDetObservationTable(['a'], sul, 15)
        close_table(ot)
        hypothesis = ot.gen_hypothesis()
        self.assertEqual(hypothesis.characterization_set, ot.E)


if __name__ == '__main__':
    unittest.main()
