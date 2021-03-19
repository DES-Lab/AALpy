from aalpy.base import SUL
import re


class RegexSUL(SUL):
    """
    An example implementation of a system under learning that can be used to learn any regex expression.
    Note that the $ is added to the expression as in this SUL only exact matches are learned.
    """
    def __init__(self, regex: str):
        super().__init__()
        self.regex = regex if regex[-1] == '$' else regex + '$'
        self.string = ""

    def pre(self):
        self.string = ""
        pass

    def post(self):
        self.string = ""
        pass

    def step(self, letter):
        """

        Args:

            letter: single element of the input alphabet

        Returns:

            Whether the current string (previous string + letter) is accepted

        """
        if letter is not None:
            self.string += str(letter)
        return True if re.match(self.regex, self.string) else False
