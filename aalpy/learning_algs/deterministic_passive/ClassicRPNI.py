import time
from bisect import insort
from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import to_automaton, createPTA, \
    check_sequence, extract_unique_sequences


class ClassicRPNI:
    def __init__(self, data, automaton_type, print_info=True):
        self.data = data
        self.automaton_type = automaton_type
        self.print_info = print_info

        pta_construction_start = time.time()
        self.root_node = createPTA(data, automaton_type)
        self.test_data = extract_unique_sequences(self.root_node, automaton_type)

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
                if not red_state.compatible_outputs(lex_min_blue):
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
                print(sequence)
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
        for i, o in blue_node.output.items():
            red_node.output[i] = o

        for i in blue_node.children.keys():
            if i in red_node.children.keys():
                self._fold_mealy(red_node.children[i], blue_node.children[i])
            else:
                red_node.children[i] = blue_node.children[i]

