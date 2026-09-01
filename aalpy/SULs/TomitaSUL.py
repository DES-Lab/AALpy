# SUL implementing the seven classical Tomita grammars, a common benchmark for automata learning.
import re

from aalpy.base import SUL


class TomitaSUL(SUL):
    """
    Tomita grammars are often used as a benchmark for automata-related challenges. Simple SUL that implements all 7
    Tomita grammars and enables their learning.
    """

    def __init__(self, tomita_level_fun: int) -> None:
        """
        Creates a SUL for a Tomita grammar.

        :param int tomita_level_fun: Number of the Tomita grammar to learn (1-7, or -3 for the negation of grammar 3).
        """
        super().__init__()
        num_fun_map = {1: tomita_1, 2: tomita_2, 3: tomita_3, 4: tomita_4, 5: tomita_5, 6: tomita_6, 7: tomita_7,
                       -3: not_tomita_3}
        assert tomita_level_fun in num_fun_map.keys()
        self.string = ""
        self.tomita_level = num_fun_map[tomita_level_fun]

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

    def step(self, letter: str) -> bool:
        """
        Appends the letter to the accumulated string and checks it against the Tomita grammar.

        :param str letter: Single element of the input alphabet.
        :return bool: Whether the current string (previous string + letter) is accepted by the grammar.
        """
        if letter is not None:
            self.string += str(letter)
        return self.tomita_level(self.string)


_not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$")


def tomita_1(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: True if word contains no "0", False otherwise.
    """
    return "0" not in word


def tomita_2(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: True if word is "10" repeated, False otherwise.
    """
    return word == "10" * (int(len(word) / 2))


def tomita_3(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: True if word does not match the Tomita 3 grammar's complement pattern, False otherwise.
    """
    if not _not_tomita_3.match(word):
        return True
    return False


def not_tomita_3(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: The negation of tomita_3(word).
    """
    return not tomita_3(word)


def tomita_4(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: True if word contains no "000", False otherwise.
    """
    return "000" not in word


def tomita_5(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: True if word has an even count of both "0" and "1", False otherwise.
    """
    return (word.count("0") % 2 == 0) and (word.count("1") % 2 == 0)


def tomita_6(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: True if the difference between the count of "0" and "1" is divisible by 3, False otherwise.
    """
    return ((word.count("0") - word.count("1")) % 3) == 0


def tomita_7(word: str) -> bool:
    """
    :param str word: Word to check.
    :return bool: True if word contains at most one occurrence of "10", False otherwise.
    """
    return word.count("10") <= 1
