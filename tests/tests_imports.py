import unittest

class ImportTest(unittest.TestCase):

    def test_imports(self):
        try:
            import aalpy.utils
            import aalpy.oracles
            import aalpy.utils
            import aalpy.SULs
            import aalpy.learning_algs
        except:
            assert False
        assert True
