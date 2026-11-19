# Data handlers/tokenizers for loading sequence data used by the Alergia algorithm.
from abc import ABC, abstractmethod


class DataHandler(ABC):
    """
    Abstract class used for data loading for Alergia algorithm. Usage of class is not needed, but recommended for
    consistency.
    """

    @abstractmethod
    def tokenize_data(self, path: str) -> list:
        """
        Tokenizes data found at the given path.

        :param str path: path to the data file.
        :return list: list of tokenized sequences.
        """
        pass


class CharacterTokenizer(DataHandler):
    """
    Used for Markov Chain data parsing.
    Processes data where each input is a single character.
    Each input sequence is in the separate line.
    """

    def tokenize_data(self, path: str) -> list[list[str]]:
        """
        Tokenizes each line of the file into a list of single characters.

        :param str path: path to the data file.
        :return list[list[str]]: list of tokenized sequences, one per line.
        """
        data = []
        lines = open(path).read().splitlines()
        for l in lines:
            data.append(list(l))
        return data


class DelimiterTokenizer(DataHandler):
    """
    Used for Markov Chain data parsing.
    Processes data where each input is separated by the delimiter.
    Each input sequence is in the separate line.
    """

    def tokenize_data(self, path: str, delimiter: str = ',') -> list[list[str]]:
        """
        Tokenizes each line of the file by splitting on the given delimiter.

        :param str path: path to the data file.
        :param str delimiter: delimiter separating inputs in a line.
        :return list[list[str]]: list of tokenized sequences, one per line.
        """
        data = []
        lines = open(path).read().splitlines()
        for l in lines:
            data.append(l.split(delimiter))
        return data


class IODelimiterTokenizer(DataHandler):
    """
    Used for Markov Decision Process data parsing.
    Processes data where each input/output is separated by the io_delimiter, and i/o pairs are separated
    by word delimiter.
    Each [output, tuple(input,output)*] sequence is in the separate line.
    """

    def tokenize_data(self, path: str, io_delimiter: str = '/', word_delimiter: str = ',') -> list[list]:
        """
        Tokenizes each line of the file into an initial output followed by (input, output) tuples.

        :param str path: path to the data file.
        :param str io_delimiter: delimiter separating an input from its output within a word.
        :param str word_delimiter: delimiter separating words (initial output and input/output pairs) in a line.
        :return list[list]: list of tokenized sequences, one per line.
        """
        data = []
        lines = open(path).read().splitlines()
        for l in lines:
            words = l.split(word_delimiter)
            seq = [words[0]]
            for w in words[1:]:
                i_o = w.split(io_delimiter)
                if len(i_o) != 2:
                    print('Data formatting error. io_delimiter should split words into <input> <delim> <output>'
                          'where <delim> is values of param \"io_delimiter\'"')
                    exit(-1)
                seq.append(tuple([try_int(i_o[0]), try_int(i_o[1])]))
            data.append(seq)
        return data


def try_int(x: str) -> int | str:
    """
    Converts a string to an int if it represents a digit, otherwise returns it unchanged.

    :param str x: string to convert.
    :return int | str: the converted integer, or the original string if not convertible.
    """
    if str.isdigit(x):
        return int(x)
    return x
