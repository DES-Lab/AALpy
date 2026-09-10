import unittest

from aalpy.learning_algs.general_passive.AssociatedData import CountData


class TestCountDataMerge(unittest.TestCase):
    def test_merge_does_not_alias_the_other_operands_dict(self):
        # merge() used to do x[in_sym] = y_o_dict for an input unseen in x, aliasing y's inner dict
        # instead of copying it, so later mutating y_o_dict would silently change x's counts too.
        x = {}
        y_o_dict = {'out1': 1}
        y = {'a': y_o_dict}
        merged = CountData.merge(x, y)
        y_o_dict['out1'] = 100
        self.assertEqual(merged['a'], {'out1': 1})


if __name__ == '__main__':
    unittest.main()
