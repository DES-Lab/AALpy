import importlib
import pkgutil
import sys
import unittest

import aalpy


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


class CircularImportTest(unittest.TestCase):
    """
    test_imports above only ever imports aalpy's submodules in one fixed order, in one process. A
    circular dependency between two modules can hide behind that: it only surfaces when whichever
    module is involved in the cycle gets imported *first*, before the module it depends on has been
    (even partially) loaded. This test instead imports every submodule of aalpy as the very first
    aalpy-related import of a clean module cache, one at a time, which is exactly the situation a
    circular import fails in.
    """

    def test_every_submodule_imports_standalone(self):
        submodules = sorted(name for _, name, _ in pkgutil.walk_packages(aalpy.__path__, prefix='aalpy.'))
        self.assertGreater(len(submodules), 0, "no aalpy submodules were discovered, the test itself is broken")

        # every other test module in the suite already imported classes (Dfa, MealyMachine, ...) from the
        # ORIGINAL aalpy modules at collection time. Re-importing here creates new, distinct class objects;
        # if left in sys.modules afterward, later tests comparing an old-class instance against a new-class
        # instance (e.g. bisimilar()'s `a1.__class__ != a2.__class__` check) would fail spuriously. So the
        # original modules must be restored once this test is done, regardless of outcome.
        original_modules = {name: module for name, module in sys.modules.items()
                            if name == 'aalpy' or name.startswith('aalpy.')}

        failures = {}
        try:
            for name in submodules:
                # drop every previously (possibly partially) loaded aalpy module so this import
                # starts from a clean slate, as if it were the first aalpy import in a fresh interpreter
                for cached in [m for m in sys.modules if m == 'aalpy' or m.startswith('aalpy.')]:
                    del sys.modules[cached]
                try:
                    importlib.import_module(name)
                except ImportError as e:
                    failures[name] = str(e)
        finally:
            for cached in [m for m in sys.modules if m == 'aalpy' or m.startswith('aalpy.')]:
                del sys.modules[cached]
            sys.modules.update(original_modules)

        self.assertEqual(failures, {}, f"modules that cannot be imported standalone (likely circular imports): "
                                       f"{failures}")
