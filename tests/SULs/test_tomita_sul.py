import unittest

from aalpy.SULs import TomitaSUL


class TestTomitaSUL(unittest.TestCase):
    def test_invalid_level_raises(self):
        with self.assertRaises(AssertionError):
            TomitaSUL(0)

    def test_pre_resets_accumulated_string(self):
        sul = TomitaSUL(1)
        sul.pre()
        sul.step('0')
        self.assertEqual(sul.string, '0')
        sul.pre()
        self.assertEqual(sul.string, '')

    def test_post_resets_accumulated_string(self):
        sul = TomitaSUL(1)
        sul.pre()
        sul.step('1')
        sul.post()
        self.assertEqual(sul.string, '')

    def test_step_none_does_not_corrupt_accumulated_string(self):
        # regression test: step() used to check `if input` (the builtin, always truthy) instead of
        # `if letter is not None`, so step(None) appended the literal text "None" to the string
        sul = TomitaSUL(1)
        sul.pre()
        result = sul.step(None)
        self.assertEqual(sul.string, '')
        self.assertTrue(result)

    def test_query_empty_word_on_tomita_1(self):
        # tomita_1 accepts the empty word (no '0's in it)
        sul = TomitaSUL(1)
        self.assertEqual(sul.query(()), [True])

    def test_tomita_1_accepts_only_words_without_zero(self):
        sul = TomitaSUL(1)
        self.assertEqual(sul.query(('1', '1', '1')), [True, True, True])
        self.assertEqual(sul.query(('1', '0', '1')), [True, False, False])

    def test_tomita_2_accepts_only_repeated_10(self):
        sul = TomitaSUL(2)
        self.assertEqual(sul.query(('1', '0', '1', '0')), [False, True, False, True])

    def test_tomita_3_and_its_negation_are_complementary(self):
        sul3 = TomitaSUL(3)
        sul_not3 = TomitaSUL(-3)
        word = ('1', '0', '0', '1', '1', '0')
        result3 = sul3.query(word)
        result_not3 = sul_not3.query(word)
        self.assertEqual(result3, [not r for r in result_not3])

    def test_tomita_4_rejects_three_consecutive_zeros(self):
        sul = TomitaSUL(4)
        self.assertEqual(sul.query(('0', '0', '0')), [True, True, False])

    def test_tomita_5_even_counts_of_both_symbols(self):
        sul = TomitaSUL(5)
        self.assertEqual(sul.query(()), [True])
        self.assertEqual(sul.query(('0', '0', '1', '1')), [False, True, False, True])

    def test_tomita_6_difference_divisible_by_three(self):
        sul = TomitaSUL(6)
        self.assertEqual(sul.query(()), [True])
        self.assertEqual(sul.query(('0', '0', '0')), [False, False, True])

    def test_tomita_7_at_most_one_descent(self):
        sul = TomitaSUL(7)
        self.assertEqual(sul.query(('1', '1', '0', '0', '1', '1')), [True, True, True, True, True, True])
        self.assertEqual(sul.query(('1', '0', '1', '0')), [True, True, True, False])


if __name__ == '__main__':
    unittest.main()
