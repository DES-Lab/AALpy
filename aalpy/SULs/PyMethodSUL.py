from aalpy.base import SUL


class FunctionDecorator:
    """
    Decorator of methods found in the SUL class.
    """

    def __init__(self, function, args=None):
        """
        Args:

            function: function of the class to be learned

            args: arguments to be passed to the function. Either a single argument, or a list of arguments if
                function has more than one parameter.
        """

        self.function = function
        self.args = None
        if args:
            self.args = [args] if not isinstance(args, list) else args

    def __repr__(self):
        if self.args:
            return f'{self.function.__name__}{self.args}'
        return self.function.__name__


class PyClassSUL(SUL):
    """
    System under learning for inferring python classes.
    """
    def __init__(self, python_class):
        """
        Args:

            python_class: class to be learned
        """
        super().__init__()
        self._class = python_class
        self.sul: object = None

    def pre(self):
        """
        Do the reset by initializing the class again or call reset method of the class
        """
        self.sul = self._class()

    def post(self):
        pass

    def step(self, letter):
        """
        Executes the function(with arguments) found in letter against the SUL

        Args:

            letter: single input of type FunctionDecorator

        Returns:

            output of the function

        """
        if letter.args:
            return getattr(self.sul, letter.function.__name__, letter)(*letter.args)
        return getattr(self.sul, letter.function.__name__, letter)()
