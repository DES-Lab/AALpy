import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Dfa, DfaState, MealyMachine, MealyState, MooreMachine, MooreState
from aalpy.learning_algs.deterministic.ObservationTable import ObservationTable
from aalpy.utils import get_Angluin_dfa
from aalpy.utils.ModelChecking import bisimilar


def three_state_dfa():
    """
    3-state minimal DFA over {a, b}. q0 is accepting; q1 and q2 are only
    distinguishable by the suffix 'b' (both go to a non-accepting state on 'a').
    q0 --a--> q1   q0 --b--> q0
    q1 --a--> q2   q1 --b--> q0
    q2 --a--> q1   q2 --b--> q2
    """
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q2 = DfaState('q2', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q2, 'b': q0}
    q2.transitions = {'a': q1, 'b': q2}
    dfa = Dfa(q0, [q0, q1, q2])
    dfa.compute_prefixes()
    return dfa


def two_state_mealy():
    """
    2-state Mealy machine over {a, b} where a self loop is only reachable from s1.
    s0 --a/1--> s1   s0 --b/0--> s0
    s1 --a/0--> s0   s1 --b/1--> s1
    """
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s0.transitions = {'a': s1, 'b': s0}
    s0.output_fun = {'a': 1, 'b': 0}
    s1.transitions = {'a': s0, 'b': s1}
    s1.output_fun = {'a': 0, 'b': 1}
    mm = MealyMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm


def two_state_moore():
    s0 = MooreState('s0', output=1)
    s1 = MooreState('s1', output=0)
    s0.transitions = {'a': s1, 'b': s0}
    s1.transitions = {'a': s0, 'b': s1}
    moore = MooreMachine(s0, [s0, s1])
    moore.compute_prefixes()
    return moore


class TestObservationTableInit(unittest.TestCase):
    def test_dfa_initial_e_set_contains_empty_word(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        self.assertEqual(table.E, [tuple()])
        self.assertEqual(table.S, [tuple()])
        self.assertEqual(table.A, [('a',), ('b',)])

    def test_moore_initial_e_set_contains_empty_word(self):
        moore = two_state_moore()
        sul = AutomatonSUL(moore)
        table = ObservationTable(['a', 'b'], sul, 'moore')
        self.assertEqual(table.E, [tuple()])

    def test_mealy_initial_e_set_is_whole_alphabet(self):
        mm = two_state_mealy()
        sul = AutomatonSUL(mm)
        table = ObservationTable(['a', 'b'], sul, 'mealy')
        self.assertEqual(table.E, [('a',), ('b',)])

    def test_invalid_automaton_type_raises(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        with self.assertRaises(AssertionError):
            ObservationTable(['a', 'b'], sul, 'not_a_type')


class TestSDotA(unittest.TestCase):
    def test_s_dot_a_excludes_elements_already_in_s(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        self.assertEqual(set(table.s_dot_a()), {('a',), ('b',)})

        table.S.append(('a',))
        self.assertEqual(set(table.s_dot_a()), {('b',), ('a', 'a'), ('a', 'b')})


class TestUpdateObsTable(unittest.TestCase):
    def test_update_fills_row_for_empty_prefix(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        self.assertEqual(table.T[tuple()], (True,))
        self.assertEqual(table.T[('a',)], (False,))
        self.assertEqual(table.T[('b',)], (True,))

    def test_update_is_idempotent_and_does_not_reissue_queries(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        queries_after_first = sul.num_queries
        table.update_obs_table()
        self.assertEqual(sul.num_queries, queries_after_first)

    def test_update_with_explicit_s_and_e_set(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.E.append(('b',))
        table.update_obs_table(s_set=[tuple()], e_set=[('b',)])
        self.assertEqual(len(table.T[tuple()]), 1)
        self.assertEqual(table.T[tuple()], (True,))

    def test_growing_e_set_requires_e_set_argument_for_new_column(self):
        """
        update_obs_table() with no arguments re-derives update_S/update_E from S/E, and for a row
        that is already partially filled it only (re)asks the *first* `len(E) - len(T[s])` columns
        of E in order -- it does not know which specific column(s) are new. Callers therefore must
        pass e_set=<newly added suffixes> when the E set grows (as LStar.py always does); calling it
        bare after extending self.E does not compute the new column correctly.
        """
        dfa = get_Angluin_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        table.E.append(('a',))
        table.update_obs_table()
        # The new column was (incorrectly, from the bare call's perspective) filled by re-querying
        # the *old* suffix again instead of the newly appended one.
        self.assertEqual(table.T[('a',)], (False, False))

        # Using the documented e_set argument computes the new column correctly instead.
        table.T.clear()
        table.update_obs_table()
        table.update_obs_table(e_set=[('a',)])
        self.assertEqual(table.T[('a',)], (False, True))


class TestGetRowsToClose(unittest.TestCase):
    def test_no_rows_to_close_returns_none_when_table_closed(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        table.S.append(('a',))
        table.S.append(('a', 'a'))
        table.update_obs_table()
        self.assertIsNone(table.get_rows_to_close('longest_first'))

    def test_shortest_first_returns_the_distinguishing_row(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        rows = table.get_rows_to_close('shortest_first')
        self.assertEqual(rows, [('a',)])

    def test_single_returns_only_one_row(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        rows = table.get_rows_to_close('single')
        self.assertEqual(rows, [('a',)])

    def _angluin_table_with_two_rows_to_close(self):
        dfa = get_Angluin_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        table.E.append(('a',))
        table.update_obs_table(e_set=[('a',)])
        return table

    def test_longest_first_and_shortest_first_find_both_distinguishing_rows(self):
        table = self._angluin_table_with_two_rows_to_close()
        self.assertEqual(set(table.get_rows_to_close('longest_first')), {('a',), ('b',)})
        self.assertEqual(set(table.get_rows_to_close('shortest_first')), {('a',), ('b',)})

    def test_single_longest_returns_a_single_row(self):
        table = self._angluin_table_with_two_rows_to_close()
        rows = table.get_rows_to_close('single_longest')
        self.assertEqual(len(rows), 1)

    def test_longest_first_orders_rows_by_decreasing_length(self):
        # Directly populate S/T (rather than driving this through a SUL) to get full control over
        # row values and prefix lengths, since get_rows_to_close only ever looks at S, A and T.
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.S = [tuple(), ('a',), ('a', 'b')]
        table.T[tuple()] = (0,)
        table.T[('a',)] = (1,)
        table.T[('a', 'b')] = (2,)
        table.T[('b',)] = (3,)
        table.T[('a', 'a')] = (4,)
        table.T[('a', 'b', 'a')] = (5,)
        table.T[('a', 'b', 'b')] = (6,)

        rows = table.get_rows_to_close('longest_first')
        self.assertEqual(set(rows), {('b',), ('a', 'a'), ('a', 'b', 'a'), ('a', 'b', 'b')})
        lengths = [len(r) for r in rows]
        self.assertEqual(lengths, sorted(lengths, reverse=True))
        self.assertEqual(len(rows[0]), 3)


class TestGetCausesOfInconsistency(unittest.TestCase):
    def test_consistent_table_returns_none(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        self.assertIsNone(table.get_causes_of_inconsistency())

    def test_inconsistency_detected_and_cause_returned(self):
        dfa = get_Angluin_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        table.E.append(('a',))
        table.update_obs_table(e_set=[('a',)])
        table.S.append(('a',))
        table.S.append(('b',))
        table.update_obs_table()

        # ('b', 'a') has the same row, (False, False), as ('b',) under the current E, but the two
        # prefixes are not actually equivalent (they reach different DFA states); adding the
        # duplicate-row prefix to S manually is what a cex-processing step would eventually force,
        # and it makes the table inconsistent since their 'b'-extensions differ.
        table.S.append(('b', 'a'))
        table.update_obs_table()

        cause = table.get_causes_of_inconsistency()
        self.assertEqual(cause, [('b',)])


class TestGenHypothesis(unittest.TestCase):
    def test_gen_hypothesis_dfa_matches_original(self):
        dfa = get_Angluin_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        table.E.append(('a',))
        table.update_obs_table(e_set=[('a',)])
        table.S.append(('a',))
        table.S.append(('b',))
        table.update_obs_table()
        table.E.append(('b',))
        table.update_obs_table(e_set=[('b',)])
        table.S.append(('a', 'b'))
        table.update_obs_table()

        hyp = table.gen_hypothesis()
        self.assertEqual(len(hyp.states), 4)
        self.assertTrue(bisimilar(dfa, hyp))

    def test_gen_hypothesis_mealy_output_fun_matches(self):
        mm = two_state_mealy()
        sul = AutomatonSUL(mm)
        table = ObservationTable(['a', 'b'], sul, 'mealy')
        table.update_obs_table()
        table.S.append(('a',))
        table.update_obs_table()

        hyp = table.gen_hypothesis()
        self.assertEqual(len(hyp.states), 2)
        for word in [('a',), ('b',), ('a', 'a'), ('b', 'a', 'b')]:
            expected = mm.execute_sequence(mm.initial_state, word)
            actual = hyp.execute_sequence(hyp.initial_state, word)
            self.assertEqual(expected, actual)

    def test_gen_hypothesis_no_cex_processing_deduplicates_rows(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        # ('a',) is genuinely new, but ('b',) is a self-loop back to q0, so its row is identical to
        # the empty prefix's row; the table is still closed since ('a',) closes the only new row.
        table.S.append(('a',))
        table.S.append(('b',))
        table.update_obs_table()
        self.assertIsNone(table.get_rows_to_close())

        hyp = table.gen_hypothesis(no_cex_processing_used=True)
        # () and ('b',) collapse into a single representative state, ('a',) remains distinct.
        self.assertEqual(len(hyp.states), 2)


class TestGetRowRepresentatives(unittest.TestCase):
    def test_representatives_prefer_shortest_prefix_per_row(self):
        dfa = three_state_dfa()
        sul = AutomatonSUL(dfa)
        table = ObservationTable(['a', 'b'], sul, 'dfa')
        table.update_obs_table()
        table.S.append(('b',))
        table.S.append(('b', 'b'))
        table.update_obs_table()

        representatives = table._get_row_representatives()
        # (), ('b',) and ('b', 'b') all share the same row (all lead to accepting q0),
        # so only the shortest, (), should be kept as representative.
        self.assertEqual(representatives, [tuple()])


if __name__ == '__main__':
    unittest.main()
