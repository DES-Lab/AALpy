import tempfile
import unittest
from pathlib import Path

from aalpy.utils.DataHandler import CharacterTokenizer, DelimiterTokenizer, IODelimiterTokenizer, try_int


class TestTryInt(unittest.TestCase):
    def test_digit_string_converted(self):
        self.assertEqual(try_int('42'), 42)
        self.assertIsInstance(try_int('42'), int)

    def test_non_digit_string_unchanged(self):
        self.assertEqual(try_int('abc'), 'abc')

    def test_negative_number_not_converted(self):
        self.assertEqual(try_int('-1'), '-1')


class TestCharacterTokenizer(unittest.TestCase):
    def test_tokenizes_each_line_into_characters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('abc\nde\n')
            data = CharacterTokenizer().tokenize_data(str(path))
            self.assertEqual(data, [['a', 'b', 'c'], ['d', 'e']])

    def test_empty_file_produces_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('')
            data = CharacterTokenizer().tokenize_data(str(path))
            self.assertEqual(data, [])


class TestDelimiterTokenizer(unittest.TestCase):
    def test_default_comma_delimiter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('a,b,c\nx,y\n')
            data = DelimiterTokenizer().tokenize_data(str(path))
            self.assertEqual(data, [['a', 'b', 'c'], ['x', 'y']])

    def test_custom_delimiter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('a;b;c\n')
            data = DelimiterTokenizer().tokenize_data(str(path), delimiter=';')
            self.assertEqual(data, [['a', 'b', 'c']])


class TestIODelimiterTokenizer(unittest.TestCase):
    def test_tokenizes_initial_output_and_io_pairs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('out0,i1/o1,i2/o2\n')
            data = IODelimiterTokenizer().tokenize_data(str(path))
            self.assertEqual(data, [['out0', ('i1', 'o1'), ('i2', 'o2')]])

    def test_digit_inputs_outputs_converted_to_int(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('0,1/2,3/4\n')
            data = IODelimiterTokenizer().tokenize_data(str(path))
            self.assertEqual(data, [['0', (1, 2), (3, 4)]])

    def test_custom_delimiters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('out0|i1:o1|i2:o2\n')
            data = IODelimiterTokenizer().tokenize_data(str(path), io_delimiter=':', word_delimiter='|')
            self.assertEqual(data, [['out0', ('i1', 'o1'), ('i2', 'o2')]])

    def test_malformed_io_word_exits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'data.txt'
            path.write_text('out0,badword\n')
            with self.assertRaises(SystemExit):
                IODelimiterTokenizer().tokenize_data(str(path))


if __name__ == '__main__':
    unittest.main()
