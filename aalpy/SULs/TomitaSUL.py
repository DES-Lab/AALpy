import re

from aalpy.base import SUL


class TomitaSUL(SUL):
    """
    Tomita grammars are often used as a benchmark for automata-related challenges. Simple SUL that implements all 7
    Tomita grammars and enables their learning.
    """

    def __init__(self, tomita_level_fun):
        super().__init__()
        num_fun_map = {1: tomita_1, 2: tomita_2, 3: tomita_3, 4: tomita_4, 5: tomita_5, 6: tomita_6, 7: tomita_7,
                       -3: not_tomita_3}
        assert tomita_level_fun in num_fun_map.keys()
        self.string = ""
        self.tomita_level = num_fun_map[tomita_level_fun]

    def pre(self):
        self.string = ""
        pass

    def post(self):
        self.string = ""
        pass

    def step(self, letter):
        if input:
            self.string += str(letter)
        return self.tomita_level(self.string)


_not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$")


def tomita_1(word):
    return "0" not in word


def tomita_2(word):
    return word == "10" * (int(len(word) / 2))


def tomita_3(word):
    if not _not_tomita_3.match(word):
        return True
    return False


def not_tomita_3(word):
    return not tomita_3(word)


def tomita_4(word):
    return "000" not in word


def tomita_5(word):
    return (word.count("0") % 2 == 0) and (word.count("1") % 2 == 0)


def tomita_6(word):
    return ((word.count("0") - word.count("1")) % 3) == 0


def tomita_7(word):
    return word.count("10") <= 1
