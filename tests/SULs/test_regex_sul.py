import unittest

from aalpy.SULs import RegexSUL


class TestRegexSUL(unittest.TestCase):
    def test_appends_trailing_dollar_if_missing(self):
        sul = RegexSUL('ab*')
        self.assertEqual(sul.regex, 'ab*$')

    def test_does_not_duplicate_trailing_dollar(self):
        sul = RegexSUL('ab*$')
        self.assertEqual(sul.regex, 'ab*$')

    def test_pre_resets_accumulated_string(self):
        sul = RegexSUL('a$')
        sul.pre()
        sul.step('a')
        self.assertEqual(sul.string, 'a')
        sul.pre()
        self.assertEqual(sul.string, '')

    def test_post_resets_accumulated_string(self):
        sul = RegexSUL('a$')
        sul.pre()
        sul.step('a')
        sul.post()
        self.assertEqual(sul.string, '')

    def test_step_none_does_not_change_string(self):
        sul = RegexSUL('a$')
        sul.pre()
        sul.step(None)
        self.assertEqual(sul.string, '')

    def test_exact_match_required(self):
        sul = RegexSUL('ab')
        self.assertEqual(sul.query(('a', 'b')), [False, True])

    def test_query_empty_word(self):
        sul = RegexSUL('a*')
        # empty string matches 'a*$'
        self.assertEqual(sul.query(()), [True])

    def test_rejects_once_pattern_cannot_match(self):
        sul = RegexSUL('ab$')
        self.assertEqual(sul.query(('b',)), [False])

    def test_accepts_intermediate_and_final_prefixes(self):
        sul = RegexSUL('a+b')
        result = sul.query(('a', 'a', 'b'))
        self.assertEqual(result, [False, False, True])


if __name__ == '__main__':
    unittest.main()
