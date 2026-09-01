import tempfile
import time
import unittest
from pathlib import Path

from aalpy.automata import NDMooreMachine, NDMooreState
from aalpy.utils import save_automaton_to_file, load_automaton_from_file, visualize_automaton
from aalpy.utils.BenchmarkSULs import get_Angluin_dfa, get_benchmark_ONFSM, get_faulty_coffee_machine_SMM
from aalpy.utils.BenchmarkSevpaModels import sevpa_for_L1
from aalpy.utils.BenchmarkVpaModels import vpa_L1


def ndmoore_machine():
    q0 = NDMooreState('q0', output='x')
    q1 = NDMooreState('q1', output='y')
    q0.transitions['a'].append(q0)
    q0.transitions['a'].append(q1)
    q1.transitions['a'].append(q1)
    return NDMooreMachine(q0, [q0, q1])


class TestSaveAutomatonToFileEdgeCases(unittest.TestCase):
    def test_unsupported_file_type_raises(self):
        dfa = get_Angluin_dfa()
        with self.assertRaises(AssertionError):
            save_automaton_to_file(dfa, path='irrelevant', file_type='bogus')

    def test_string_file_type_returns_dot_string_without_writing_file(self):
        dfa = get_Angluin_dfa()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'model'
            result = save_automaton_to_file(dfa, path=str(path), file_type='string')
            self.assertIsInstance(result, str)
            self.assertIn('digraph', result)
            self.assertFalse(path.with_suffix('.string').exists())

    def test_string_output_contains_all_state_ids(self):
        dfa = get_Angluin_dfa()
        result = save_automaton_to_file(dfa, file_type='string')
        for state in dfa.states:
            self.assertIn(state.state_id, result)


class TestSaveLoadAdditionalAutomatonTypes(unittest.TestCase):
    def test_ndmoore_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'ndmoore'
            model = ndmoore_machine()
            model.save(str(path))
            loaded = load_automaton_from_file(path.with_suffix('.dot'), automaton_type='ndmoore')
            self.assertEqual(loaded.size, model.size)
            self.assertEqual(set(loaded.get_input_alphabet()), set(model.get_input_alphabet()))
            for state in loaded.states:
                self.assertIsNotNone(state.output)

    def test_sevpa_roundtrip_preserves_alphabet_and_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'sevpa'
            model = sevpa_for_L1()
            model.save(str(path))
            loaded = load_automaton_from_file(path.with_suffix('.dot'), automaton_type='sevpa')
            self.assertEqual(loaded.size, model.size)
            self.assertEqual(set(model.get_input_alphabet().get_merged_alphabet()),
                             set(loaded.get_input_alphabet().get_merged_alphabet()))
            self.assertEqual({s.state_id for s in model.states if s.is_accepting},
                             {s.state_id for s in loaded.states if s.is_accepting})

    def test_vpa_roundtrip_skips_error_sink_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'vpa'
            model = vpa_L1()
            non_sink_states = [s for s in model.states if s.state_id != 'ErrorSinkState']
            model.save(str(path))
            loaded = load_automaton_from_file(path.with_suffix('.dot'), automaton_type='vpa')
            self.assertEqual(loaded.size, len(non_sink_states))
            self.assertEqual(set(model.get_input_alphabet().get_merged_alphabet()),
                             set(loaded.get_input_alphabet().get_merged_alphabet()))

    def test_onfsm_roundtrip_preserves_transitions_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'onfsm'
            model = get_benchmark_ONFSM()
            model.save(str(path))
            loaded = load_automaton_from_file(path.with_suffix('.dot'), automaton_type='onfsm')
            self.assertEqual(loaded.size, model.size)
            for state in loaded.states:
                for i in loaded.get_input_alphabet():
                    self.assertGreater(len(state.transitions[i]), 0)

    def test_smm_roundtrip_preserves_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'smm'
            model = get_faulty_coffee_machine_SMM()
            model.save(str(path))
            loaded = load_automaton_from_file(path.with_suffix('.dot'), automaton_type='smm')
            for orig_state, loaded_state in zip(model.states, loaded.states):
                for i in model.get_input_alphabet():
                    orig_probs = sorted(p for _, _, p in orig_state.transitions[i])
                    loaded_probs = sorted(p for _, _, p in loaded_state.transitions[i])
                    self.assertEqual(orig_probs, loaded_probs)


class TestLoadMalformedDotFile(unittest.TestCase):
    def test_missing_start_state_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'bad.dot'
            path.write_text('digraph g {\nq0 [label="q0"];\nq0 -> q0 [label="a"];\n}\n')
            with self.assertRaises(AssertionError):
                load_automaton_from_file(path, automaton_type='dfa')

    def test_start_state_pointing_to_undefined_state_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'bad.dot'
            path.write_text(
                'digraph g {\n'
                '__start0 [label="", shape=none];\n'
                '__start0 -> qX;\n'
                'q0 [label="q0"];\n'
                'q0 -> q0 [label="a"];\n'
                '}\n'
            )
            with self.assertRaises(AssertionError):
                load_automaton_from_file(path, automaton_type='dfa')


class TestVisualizeAutomaton(unittest.TestCase):
    def test_visualize_writes_file_in_background_thread(self):
        dfa = get_Angluin_dfa()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'viz'
            visualize_automaton(dfa, path=str(path), file_type='dot')
            deadline = time.time() + 5
            while time.time() < deadline and not path.with_suffix('.dot').exists():
                time.sleep(0.05)
            self.assertTrue(path.with_suffix('.dot').exists())

    def test_visualize_large_automaton_prints_warning(self):
        # automaton_types with >= 25 states trigger an extra warning print; just make sure it does
        # not raise and still produces the file.
        from aalpy.utils import generate_random_dfa
        import random
        random.seed(42)
        dfa = generate_random_dfa(num_states=26, alphabet=['a', 'b'], ensure_minimality=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'viz_large'
            visualize_automaton(dfa, path=str(path), file_type='dot')
            deadline = time.time() + 5
            while time.time() < deadline and not path.with_suffix('.dot').exists():
                time.sleep(0.05)
            self.assertTrue(path.with_suffix('.dot').exists())


if __name__ == '__main__':
    unittest.main()
