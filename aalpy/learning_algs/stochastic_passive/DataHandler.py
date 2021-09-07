from abc import ABC, abstractmethod

from aalpy.SULs import MdpSUL
from aalpy.utils import generate_random_mdp, visualize_automaton
from random import seed, choice, random, randint


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


if __name__ == '__main__':
    seed(4)
    mdp, inps = generate_random_mdp(4, 2, custom_outputs=['A','B','C','D'])
    visualize_automaton(mdp, path='Original')
    sul = MdpSUL(mdp)
    inputs = mdp.get_input_alphabet()

    data = []
    for _ in range(10000):
        str_len = randint(5,12)
        seq = [sul.pre()]
        for _ in range(str_len):
            i = choice(inputs)
            o = sul.step(i)
            seq.append((i, o))
        sul.post()
        data.append(seq)

    with open('mdpData.txt', 'w') as file:
        for seq in data:
            sting = f'{seq[0]},'
            for io in seq[1:]:
                sting += f'{io[0]}/{io[1]},'

            file.write(f'{sting[:-1]}\n')

    file.close()