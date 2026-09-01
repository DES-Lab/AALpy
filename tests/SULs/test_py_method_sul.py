import unittest

from aalpy.SULs.PyMethodSUL import FunctionDecorator, PyClassSUL


class Counter:
    """Tiny stateful class used as a Python-class SUL target."""

    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def add(self, amount):
        self.value += amount
        return self.value

    def get(self):
        return self.value


class TestFunctionDecorator(unittest.TestCase):
    def test_repr_without_args(self):
        decorator = FunctionDecorator(Counter.increment)
        self.assertEqual(repr(decorator), 'increment')

    def test_repr_with_single_arg(self):
        decorator = FunctionDecorator(Counter.add, args=5)
        self.assertEqual(repr(decorator), 'add[5]')

    def test_repr_with_multiple_args(self):
        decorator = FunctionDecorator(Counter.add, args=[1, 2])
        self.assertEqual(repr(decorator), 'add[1, 2]')

    def test_no_args_when_none_given(self):
        decorator = FunctionDecorator(Counter.increment, args=None)
        self.assertIsNone(decorator.args)


class TestPyClassSUL(unittest.TestCase):
    def test_pre_creates_a_fresh_instance(self):
        sul = PyClassSUL(Counter)
        sul.pre()
        first_instance = sul.sul
        sul.step(FunctionDecorator(Counter.increment))
        sul.pre()
        second_instance = sul.sul

        self.assertIsNot(first_instance, second_instance)
        self.assertEqual(second_instance.value, 0)

    def test_step_calls_method_without_args(self):
        sul = PyClassSUL(Counter)
        sul.pre()
        result = sul.step(FunctionDecorator(Counter.increment))
        self.assertEqual(result, 1)

    def test_step_calls_method_with_args(self):
        sul = PyClassSUL(Counter)
        sul.pre()
        result = sul.step(FunctionDecorator(Counter.add, args=5))
        self.assertEqual(result, 5)

    def test_state_persists_across_steps_within_one_pre(self):
        sul = PyClassSUL(Counter)
        sul.pre()
        sul.step(FunctionDecorator(Counter.increment))
        sul.step(FunctionDecorator(Counter.increment))
        result = sul.step(FunctionDecorator(Counter.get))
        self.assertEqual(result, 2)

    def test_query_runs_sequence_of_function_calls(self):
        sul = PyClassSUL(Counter)
        result = sul.query((FunctionDecorator(Counter.increment), FunctionDecorator(Counter.add, args=3)))
        self.assertEqual(result, [1, 4])


if __name__ == '__main__':
    unittest.main()
