import unittest


class ImportTest(unittest.TestCase):

    def test_imports(self):
        try:
            import aalpy.utils
            import aalpy.oracles
            import aalpy.utils
            import aalpy.SULs
            import aalpy.learning_algs
            import aalpy.base
            import aalpy.base.Automaton
            import aalpy.utils.HelperFunctions
            import aalpy.utils.DataHandler
            import aalpy.utils.AutomatonGenerators
            import aalpy.utils.ModelChecking
            import aalpy.utils.FileHandler
        except:
            assert False
        assert True
