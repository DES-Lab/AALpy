import time
from bisect import insort
from typing import Union

from aalpy.base import DeterministicAutomaton
from aalpy.learning_algs.deterministic_passive.GeneralizedStateMerging import GeneralizedStateMerging
from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import to_automaton, createPTA, \
    check_sequence, extract_unique_sequences


class RPNI:
    def __init__(self, data, automaton_type, print_info=True):
        self.data = data
        self.automaton_type = automaton_type
        self.print_info = print_info

        pta_construction_start = time.time()
        self.root_node = createPTA(data, automaton_type)
        self.test_data = extract_unique_sequences(self.root_node)

        if self.print_info:
            print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')

    def run_rpni(self):
        start_time = time.time()

        red = [self.root_node]
        blue = list(red[0].children.values())
        while blue:
            lex_min_blue = min(list(blue))
            merged = False

            for red_state in red:
                if not self._compatible_states(red_state, lex_min_blue):
                    continue
                merge_candidate = self._merge(red_state, lex_min_blue, copy_nodes=True)
                if self._compatible(merge_candidate):
                    self._merge(red_state, lex_min_blue)
                    merged = True
                    break

            if not merged:
                insort(red, lex_min_blue)
                if self.print_info:
                    print(f'\rCurrent automaton size: {len(red)}', end="")

            blue.clear()
            for r in red:
                for c in r.children.values():
                    if c not in red:
                        blue.append(c)

        if self.print_info:
            print(f'\nRPNI Learning Time: {round(time.time() - start_time, 2)}')
            print(f'RPNI Learned {len(red)} state automaton.')

        assert sorted(red, key=lambda x: len(x.prefix)) == red
        return to_automaton(red, self.automaton_type)

    def _compatible(self, root_node):
        """
        Check if current model is compatible with the data.
        """
        for sequence in self.test_data:
            if not check_sequence(root_node, sequence, automaton_type=self.automaton_type):
                return False
        return True

    def _compatible_states(self, red_node, blue_node):
        """
        Only allow merging of states that have same output(s).
        """
        if self.automaton_type != 'mealy':
            # None is compatible with everything
            return red_node.output == blue_node.output or red_node.output is None or blue_node.output is None
        else:
            red_io = {i: o for i, o in red_node.children.keys()}
            blue_io = {i: o for i, o in blue_node.children.keys()}
            for common_i in set(red_io.keys()).intersection(blue_io.keys()):
                if red_io[common_i] != blue_io[common_i]:
                    return False
        return True

    def _merge(self, red_node, lex_min_blue, copy_nodes=False):
        """
        Merge two states and return the root node of resulting model.
        """
        root_node = self.root_node.copy() if copy_nodes else self.root_node
        lex_min_blue = lex_min_blue.copy() if copy_nodes else lex_min_blue

        red_node_in_tree = root_node
        for p in red_node.prefix:
            red_node_in_tree = red_node_in_tree.children[p]

        to_update = root_node
        for p in lex_min_blue.prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[lex_min_blue.prefix[-1]] = red_node_in_tree

        if self.automaton_type != 'mealy':
            self._fold(red_node_in_tree, lex_min_blue)
        else:
            self._fold_mealy(red_node_in_tree, lex_min_blue)

        return root_node

    def _fold(self, red_node, blue_node):
        # Change the output of red only to concrete output, ignore None
        red_node.output = blue_node.output if blue_node.output is not None else red_node.output

        for i in blue_node.children.keys():
            if i in red_node.children.keys():
                self._fold(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i]

    def _fold_mealy(self, red_node, blue_node):
        blue_io_map = {i: o for i, o in blue_node.children.keys()}

        updated_keys = {}
        for io, val in red_node.children.items():
            o = blue_io_map[io[0]] if io[0] in blue_io_map.keys() else io[1]
            updated_keys[(io[0], o)] = val

        red_node.children = updated_keys

        for io in blue_node.children.keys():
            if io in red_node.children.keys():
                self._fold_mealy(red_node.children[io], blue_node.children[io])
            else:
                red_node.children[io] = blue_node.children[io]


def run_RPNI(data, automaton_type, algorithm='gsm',
             input_completeness=None, print_info=True) -> Union[DeterministicAutomaton, None]:
    """
    Run RPNI, a deterministic passive model learning algorithm.
    Resulting model conforms to the provided data.
    For more information on RPNI, check out AALpy' Wiki:
    https://github.com/DES-Lab/AALpy/wiki/RPNI---Passive-Deterministic-Automata-Learning

    Args:

        data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
        automaton_type: either 'dfa', 'mealy', 'moore'. Note that for 'mealy' machine learning, data has to be prefix-closed.
        algorithm: either 'gsm' (generalized state merging) or 'classic' for base RPNI implementation. GSM is much faster and less resource intensive.
        input_completeness: either None, 'sink_state', or 'self_loop'. If None, learned model could be input incomplete,
        sink_state will lead all undefined inputs form some state to the sink state, whereas self_loop will simply create
        a self loop. In case of Mealy learning output of the added transition will be 'epsilon'.
        print_info: print learning progress and runtime information

    Returns:

        Model conforming to the data, or None if data is non-deterministic.
    """
    assert algorithm in {'gsm', 'classic'}
    assert automaton_type in {'dfa', 'mealy', 'moore'}
    assert input_completeness in {None, 'self_loop', 'sink_state'}

    if algorithm == 'classic':
        rpni = RPNI(data, automaton_type, print_info)

        if rpni.root_node is None:
            print('Data provided to RPNI is not deterministic. Ensure that the data is deterministic, '
                  'or consider using Alergia.')
            return None
    else:
        rpni = GeneralizedStateMerging(data, automaton_type, print_info)

        if rpni.root is None:
            print('Data provided to RPNI is not deterministic. Ensure that the data is deterministic, '
                  'or consider using Alergia.')
            return None

    learned_model = rpni.run_rpni()

    if not learned_model.is_input_complete():
        if not input_completeness:
            if print_info:
                print('Warning: Learned Model is not input complete (inputs not defined for all states). '
                      'Consider calling .make_input_complete()')
        else:
            if print_info:
                print(f'Learned model was not input complete. Adapting it with {input_completeness} transitions.')
            learned_model.make_input_complete(input_completeness)

    return learned_model
