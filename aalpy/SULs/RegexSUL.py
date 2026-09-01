# SUL for learning a regular expression as a DFA-like acceptor.
import re

from aalpy.base import SUL


class RegexSUL(SUL):
    """
    An example implementation of a system under learning that can be used to learn any regex expression.
    Note that the $ is added to the expression as in this SUL only exact matches are learned.
    """
    def __init__(self, regex: str) -> None:
        """
        Creates a SUL for a regular expression.

        :param str regex: The regular expression to learn. A trailing '$' is added if missing.
        """
        super().__init__()
        self.regex = regex if regex[-1] == '$' else regex + '$'
        self.string = ""

    def pre(self) -> None:
        """
        Resets the accumulated input string.
        """
        self.string = ""
        pass

    def post(self) -> None:
        """
        Resets the accumulated input string.
        """
        self.string = ""
        pass

    def step(self, letter: str | None) -> bool:
        """
        Appends the letter to the accumulated string and checks whether it matches the regex.

        :param str | None letter: Single element of the input alphabet.
        :return bool: Whether the current string (previous string + letter) is accepted.
        """
        if letter is not None:
            self.string += str(letter)
        return True if re.match(self.regex, self.string) else False
