from abc import ABC, abstractmethod


class DataHandler(ABC):

    @abstractmethod
    def tokenize_data(self, path):
        pass


class CharacterTokenizer(DataHandler):

    def tokenize_data(self, path):
        data = []
        lines = open(path).read().splitlines()
        for l in lines:
            data.append(list(l))
        return data


class DelimiterTokenizer(DataHandler):

    def tokenize_data(self, path, delimiter=','):
        data = []
        lines = open(path).read().splitlines()
        for l in lines:
            data.append(l.split(delimiter))
        return data


class IODelimiterTokenizer(DataHandler):

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
                seq.append(tuple([i_o[0], i_o[1]]))
            data.append(seq)
        return data
