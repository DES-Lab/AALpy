import unittest

from aalpy.learning_algs.resetless.hW_datastructures import HomingSequenceIndex, ModelState


class ModelStateTests(unittest.TestCase):

    def test_new_state_has_empty_bookkeeping(self):
        state = ModelState(('a', 'b'))
        self.assertEqual(state.hs, ('a', 'b'))
        self.assertEqual(state.state_w_values, {})
        self.assertEqual(state.transitions, {})
        self.assertEqual(state.output_fun, {})
        self.assertEqual(state.transition_w_values, {})
        self.assertEqual(len(state.learned_w_per_input), 0)

    def test_learned_w_per_input_auto_creates_set_per_key(self):
        state = ModelState(())
        state.learned_w_per_input['a'].add(('w',))
        self.assertEqual(state.learned_w_per_input['a'], {('w',)})
        # accessing an unseen key must not pollute any other key's set
        self.assertEqual(state.learned_w_per_input['b'], set())
        state.learned_w_per_input['b'].add(('other',))
        self.assertEqual(state.learned_w_per_input['a'], {('w',)})

    def test_learned_w_per_input_is_independent_per_instance(self):
        # a shared mutable default here would make every ModelState alias the
        # same per-input sets, silently corrupting unrelated states
        state1 = ModelState(('x',))
        state2 = ModelState(('y',))
        state1.learned_w_per_input['a'].add(('w',))
        self.assertEqual(state2.learned_w_per_input['a'], set())


class HomingSequenceIndexTests(unittest.TestCase):

    def test_empty_h_never_flags_nondeterminism(self):
        index = HomingSequenceIndex()
        trace = [('a', 'x'), ('a', 'x'), ('a', 'y')]
        self.assertIsNone(index.scan(trace, h=()))
        self.assertEqual(index.continuation_starts(('x',)), ())

    def test_reset_clears_pairs_and_advances_scan_position(self):
        index = HomingSequenceIndex()
        trace = [('a', 'x'), ('b', '1'), ('a', 'x'), ('b', '1')]
        index.scan(trace, h=('a',))
        self.assertNotEqual(index.continuation_starts(('x',)), ())

        index.reset(trace_len=len(trace))
        self.assertEqual(index.continuation_starts(('x',)), ())
        self.assertEqual(index._scan_pos, len(trace))
        self.assertEqual(index._next_occ_min_start, len(trace))
        self.assertEqual(index._pair_progress, {})

    def test_scan_is_none_when_continuations_after_same_response_agree(self):
        index = HomingSequenceIndex()
        # h='a' occurs at position 0 and 2, both with response 'x'; both
        # continuations produce the same ('b', '1') step, so h stays consistent
        trace = [('a', 'x'), ('b', '1'), ('a', 'x'), ('b', '1')]
        self.assertIsNone(index.scan(trace, h=('a',)))
        self.assertEqual(list(index.continuation_starts(('x',))), [1, 3])

    def test_scan_detects_output_divergence_between_same_response_occurrences(self):
        index = HomingSequenceIndex()
        # h='a' occurs at position 0 and 2, both with response 'x', but the
        # continuations disagree on the output of 'b' ('1' vs '2')
        trace = [('a', 'x'), ('b', '1'), ('a', 'x'), ('b', '2')]
        extension = index.scan(trace, h=('a',))
        self.assertEqual(extension, ('b',))

    def test_scan_returns_none_for_pairs_with_different_response(self):
        index = HomingSequenceIndex()
        # h='a' occurs twice but with different responses ('x' vs 'y'), so the
        # two occurrences are never paired and no divergence can be reported
        trace = [('a', 'x'), ('b', '1'), ('a', 'y'), ('b', '2')]
        self.assertIsNone(index.scan(trace, h=('a',)))

    def test_scan_deletes_pair_once_inputs_diverge_without_reporting(self):
        index = HomingSequenceIndex()
        # continuations diverge on the *input* executed, not the output, so this
        # is not h-non-determinism and must not be reported
        trace = [('a', 'x'), ('b', '1'), ('a', 'x'), ('c', '1')]
        self.assertIsNone(index.scan(trace, h=('a',)))
        # the pair was dropped, so a later matching continuation does not
        # magically resurrect a stale comparison
        self.assertEqual(index._pair_progress, {})

    def test_self_overlapping_occurrence_is_skipped_unless_forced(self):
        index = HomingSequenceIndex()
        # h = ('a', 'a') inside a run of four 'a's: occurrences start at 0, 1, 2
        # (all matching), but the one at 1 overlaps the just-registered occurrence
        # at 0 (its continuation at 3 falls before the next allowed start) and is
        # skipped as incidental; 0 and 2 are far enough apart to both register
        trace = [('a', 'x'), ('a', 'y'), ('a', 'x'), ('a', 'y')]
        h = ('a', 'a')

        index.scan(trace, h)
        starts = sorted(c for starts in index._hs_cont_starts.values() for c in starts)
        self.assertEqual(starts, [2, 4])
        self.assertNotIn(3, starts)

    def test_forced_cont_start_registers_otherwise_skipped_occurrence(self):
        index = HomingSequenceIndex()
        trace = [('a', 'x'), ('a', 'y'), ('a', 'x'), ('a', 'y')]
        h = ('a', 'a')

        # continuation at position 3 belongs to the overlapping occurrence
        # starting at 1, which would normally be skipped as incidental
        index.scan(trace, h, forced_cont_start=3)
        starts = sorted(c for starts in index._hs_cont_starts.values() for c in starts)
        self.assertEqual(starts, [2, 3])

    def test_continuation_starts_returns_empty_tuple_for_unknown_response(self):
        index = HomingSequenceIndex()
        index.scan([('a', 'x'), ('b', '1'), ('a', 'x'), ('b', '1')], h=('a',))
        self.assertEqual(index.continuation_starts(('never', 'seen')), ())

    def test_incremental_scan_only_processes_newly_added_trace(self):
        index = HomingSequenceIndex()
        h = ('a',)
        trace = [('a', 'x'), ('b', '1')]
        self.assertIsNone(index.scan(trace, h))

        trace = trace + [('a', 'x'), ('b', '2')]
        extension = index.scan(trace, h)
        self.assertEqual(extension, ('b',))


if __name__ == '__main__':
    unittest.main()
