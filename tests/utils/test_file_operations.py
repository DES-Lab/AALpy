import tempfile
import unittest
from pathlib import Path

from aalpy.utils import generate_random_markov_chain, load_automaton_from_file
from aalpy.utils.BenchmarkSULs import get_Angluin_dfa, get_benchmark_ONFSM, get_faulty_coffee_machine_SMM, \
    get_small_pomdp
from aalpy.utils.ModelChecking import bisimilar

DOT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / 'DotModels'


def dump_and_load(model, automaton_type, directory):
    path = directory / 'model'
    model.save(str(path))
    return load_automaton_from_file(path.with_suffix('.dot'), automaton_type=automaton_type)


class TestFileHandler(unittest.TestCase):

    def test_saving_loading_roundtrip_preserves_alphabet_and_char_set(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            type_model_pairs = [
                ("dfa", get_Angluin_dfa()),
                ("mealy", load_automaton_from_file(DOT_MODELS_DIR / 'Angluin_Mealy.dot', automaton_type='mealy')),
                ("moore", load_automaton_from_file(DOT_MODELS_DIR / 'Angluin_Moore.dot', automaton_type='moore')),
                ("onfsm", get_benchmark_ONFSM()),
                ("mdp", get_small_pomdp()),
                ("mdp", load_automaton_from_file(DOT_MODELS_DIR / 'MDPs/first_grid.dot', automaton_type='mdp')),
                ("smm", get_faulty_coffee_machine_SMM()),
                ("mc", generate_random_markov_chain(num_states=10)),
            ]

            for automaton_type, model in type_model_pairs:
                loaded_model = dump_and_load(model, automaton_type, tmp_dir)
                loaded_model_twice = dump_and_load(loaded_model, automaton_type, tmp_dir)

                if automaton_type != 'mc':
                    self.assertEqual(set(model.get_input_alphabet()), set(loaded_model.get_input_alphabet()),
                                     msg=f'{automaton_type}: alphabet changed after one dump/load cycle')
                    self.assertEqual(set(model.get_input_alphabet()), set(loaded_model_twice.get_input_alphabet()),
                                     msg=f'{automaton_type}: alphabet changed after two dump/load cycles')
                else:
                    self.assertEqual(model.size, loaded_model.size)
                    self.assertEqual(model.size, loaded_model_twice.size)

                if automaton_type in {'dfa', 'moore', 'mealy'}:
                    self.assertEqual(model.compute_characterization_set(),
                                     loaded_model_twice.compute_characterization_set(),
                                     msg=f'{automaton_type}: characterization set changed after dump/load cycles')

    def test_dfa_dump_load_dump_load_is_bisimilar_to_original(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            model = get_Angluin_dfa()
            loaded_once = dump_and_load(model, 'dfa', tmp_dir)
            loaded_twice = dump_and_load(loaded_once, 'dfa', tmp_dir)

            self.assertTrue(bisimilar(model, loaded_once))
            self.assertTrue(bisimilar(model, loaded_twice))
            self.assertTrue(bisimilar(loaded_once, loaded_twice))

    def test_mealy_dump_load_dump_load_is_bisimilar_to_original(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            model = load_automaton_from_file(DOT_MODELS_DIR / 'Angluin_Mealy.dot', automaton_type='mealy')
            loaded_once = dump_and_load(model, 'mealy', tmp_dir)
            loaded_twice = dump_and_load(loaded_once, 'mealy', tmp_dir)

            self.assertTrue(bisimilar(model, loaded_once))
            self.assertTrue(bisimilar(model, loaded_twice))

    def test_moore_dump_load_dump_load_is_bisimilar_to_original(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            model = load_automaton_from_file(DOT_MODELS_DIR / 'Angluin_Moore.dot', automaton_type='moore')
            loaded_once = dump_and_load(model, 'moore', tmp_dir)
            loaded_twice = dump_and_load(loaded_once, 'moore', tmp_dir)

            self.assertTrue(bisimilar(model, loaded_once))
            self.assertTrue(bisimilar(model, loaded_twice))


if __name__ == '__main__':
    unittest.main()
