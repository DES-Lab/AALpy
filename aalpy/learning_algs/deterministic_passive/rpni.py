import random
import time
from bisect import insort

from aalpy.SULs import MooreSUL, MealySUL
from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import to_automaton, createPTA, \
    check_sequance_dfa_and_moore, check_sequance_mealy, test_rpni_with_eq_oracle, extract_unique_sequences
from aalpy.utils import load_automaton_from_file


class RPNI:
    def __init__(self, data, automaton_type, print_info=True):
        self.data = data
        self.automaton_type = automaton_type
        pta_construction_start = time.time()
        self.root_node = createPTA(data, automaton_type)
        self.test_data = extract_unique_sequences(self.root_node)
        if print_info:
            print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')
        self.print_info = print_info

    def run_rpni(self):
        start_time = time.time()

        red = [self.root_node]
        blue = list(red[0].children.values())

        while blue:
            lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))
            merged = False

            for red_state in red:
                merge_candidate = self._merge(red_state, lex_min_blue, copy_nodes=True)
                if self._compatible(merge_candidate):
                    self._merge(red_state, lex_min_blue)
                    merged = True
                    break  # I am not sure about this

            if not merged:
                insort(red, lex_min_blue)
                print(len(red))

            blue.clear()
            for r in red:
                for c in r.children.values():
                    if c not in red:
                        blue.append(c)

        if self.print_info:
            print(f'RPNI Learning Time: {round(time.time() - start_time, 2)}')
            print(f'RPNI Learned {len(red)} state automaton.')

        assert sorted(red, key=lambda x: len(x.prefix)) == red
        return to_automaton(red, self.automaton_type)

    def _compatible(self, r):
        for sequence in self.test_data:
            if self.automaton_type != 'mealy':
                sequence_passing = check_sequance_dfa_and_moore(r, sequence)
            else:
                sequence_passing = check_sequance_mealy(r, sequence)
            if not sequence_passing:
                return False
        return True

    def _merge(self, r, lex_min_blue, copy_nodes=False):
        root_node = self.root_node.copy() if copy_nodes else self.root_node
        lex_min_blue = lex_min_blue.copy() if copy_nodes else lex_min_blue

        red_node = root_node
        for p in r.prefix:
            red_node = red_node.children[p]

        b_prefix = lex_min_blue.prefix
        to_update = root_node
        for p in b_prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[b_prefix[-1]] = red_node
        self._fold(red_node, lex_min_blue)

        return root_node

    def _fold(self, red_node, blue_node):
        red_node.output = blue_node.output

        if self.automaton_type == 'mealy':
            updated_keys = {}
            for io, val in red_node.children.items():
                updated_keys[(io[0], blue_node.output)] = val
            red_node.children = updated_keys

        for i in blue_node.children.keys():
            if i in red_node.children.keys():
                self._fold(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i].copy()


def run_RPNI(data, automaton_type):
    assert automaton_type in {'dfa', 'mealy', 'moore'}
    return RPNI(data, automaton_type).run_rpni()


if __name__ == '__main__':
    # from random import seed
    # seed(5)
    # test_rpni_with_eq_oracle()
    # exit()
    #
    # random.seed(4)

    dfa = load_automaton_from_file('original.dot', automaton_type='mealy')
    #dfa = load_automaton_from_file('../../../DotModels/mooreModel.dot', automaton_type='moore')
    dfa_sul = MealySUL(dfa)
    input_al = dfa.get_input_alphabet()
    data = []
    for _ in range(1000):
        dfa_sul.pre()
        seq = []
        for _ in range(10, 20):
            i = random.choice(input_al)
            o = dfa_sul.step(i)
            seq.append((i, o))
        dfa_sul.post()
        data.append(seq)

    # data = [[('a', False), ('a', False), ('a', True)],
    #         [('a', False), ('a', False), ('b', False), ('a', True)],
    #         [('b', False), ('b', False), ('a', True)],
    #         [('b', False), ('b', False), ('a', True), ('b', False), ('a', True)],
    #         [('a', False,), ('b', False,), ('a', False)]]

    # a,bb,aab,aba
    import cProfile

    pr = cProfile.Profile()
    pr.enable()
    model = run_RPNI(data, 'moore')
    pr.disable()
    pr.print_stats(sort='tottime')

    model.visualize()
