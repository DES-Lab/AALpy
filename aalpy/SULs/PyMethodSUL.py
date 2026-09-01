# SUL for learning the behavior of an arbitrary Python class through its methods.
from typing import Any

from aalpy.base import SUL


class FunctionDecorator:
    """
    Decorator of methods found in the SUL class.
    """

    def __init__(self, function: Any, args: Any = None) -> None:
        """
        Creates a function decorator.

        :param Any function: Function of the class to be learned.
        :param Any args: Arguments to be passed to the function. Either a single argument, or a list of arguments
            if the function has more than one parameter.
        """

        self.function = function
        self.args = None
        if args:
            self.args = [args] if not isinstance(args, (list, tuple)) else args

    def __repr__(self) -> str:
        """
        :return str: A string representation of the function call.
        """
        if self.args:
            return f'{self.function.__name__}{self.args}'
        return self.function.__name__


class PyClassSUL(SUL):
    """
    System under learning for inferring python classes.
    """
    def __init__(self, python_class: type) -> None:
        """
        Creates a SUL for a Python class.

        :param type python_class: Class to be learned.
        """
        super().__init__()
        self._class = python_class
        self.sul: object = None

    def pre(self) -> None:
        """
        Do the reset by initializing the class again or call reset method of the class.
        """
        self.sul = self._class()

    def post(self) -> None:
        """
        Performs no additional cleanup, as a fresh instance is created on every pre() call.
        """
        pass

    def step(self, letter: FunctionDecorator) -> Any:
        """
        Executes the function(with arguments) found in letter against the SUL.

        :param FunctionDecorator letter: Single input of type FunctionDecorator.
        :return Any: Output of the function.
        """
        if letter.args:
            return getattr(self.sul, letter.function.__name__, letter)(*letter.args)
        return getattr(self.sul, letter.function.__name__, letter)()
