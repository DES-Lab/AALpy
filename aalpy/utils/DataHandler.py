from abc import ABC, abstractmethod


class DataHandler(ABC):
    """
    Abstract class used for data loading for Alergia algorithm. Usage of class is not needed, but recommended for
    consistency.
    """

    @abstractmethod
    def tokenize_data(self, path):
        pass


class CharacterTokenizer(DataHandler):
    """
    Used for Markov Chain data parsing.
    Processes data where each input is a single character.
    Each input sequence is in the separate line.
    """

    def tokenize_data(self, path):
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

    def tokenize_data(self, path, delimiter=','):
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

    def tokenize_data(self, path, io_delimiter='/', word_delimiter=','):
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


def try_int(x):
    if str.isdigit(x):
        return int(x)
    return x
