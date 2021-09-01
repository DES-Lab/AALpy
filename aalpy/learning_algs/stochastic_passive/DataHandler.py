from abc import ABC, abstractmethod


class DataHandler(ABC):

    @abstractmethod
    def tokenize_data(self, path):
        pass


class CharacterTokenizer(DataHandler):

    def tokenize_data(self, path):
        data = []
        with open(path) as fp:
            line = fp.readline()
            data.append(list(line))
        return data


class DelimiterTokenizer(DataHandler):

    def tokenize_data(self, path, delimiter=','):
        data = []
        with open(path) as fp:
            line = fp.readline()
            data.append(line.split(delimiter))
        return data
