import time
from bisect import insort

from aalpy.learning_algs.vpa_passive.VpaPTA import create_Vpa_PTA, extract_unique_sequences, to_vpa, visualize_vpa_pta, \
    check_vpa_sequence


class RPNI_VPA:
    def __init__(self, data, vpa_alphabet, print_info=True):
        self.data = data
        self.vpa_alphabet = vpa_alphabet
        self.print_info = print_info

        pta_construction_start = time.time()
        self.root_node = create_Vpa_PTA(data, vpa_alphabet)
        self.test_data = extract_unique_sequences(self.root_node)

        if self.print_info:
            print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')

    def run_vpa_rpni(self):
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
                if merge_candidate and self._compatible(merge_candidate):
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
        # visualize_vpa_pta(self.root_node)
        x = self._compatible(self.root_node)
        print(x)
        return to_vpa(red, self.vpa_alphabet)

    def _compatible(self, root_node):
        """
        Check if current model is compatible with the data.
        """
        for sequence in self.test_data:
            if not check_vpa_sequence(root_node, sequence, self.vpa_alphabet):
                return False
        return True

    def _compatible_states(self, red_node, blue_node):
        """
        Only allow merging of states that have same output(s), and all common push/pop pairs match
        """
        #
        return (red_node.output == blue_node.output or red_node.output is None or blue_node.output is None) \
               and all(red_node.top_of_stack[k] == blue_node.top_of_stack[k]
                       for k in red_node.top_of_stack if
                       k in blue_node.top_of_stack)

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

        fold_successful = self._fold(red_node_in_tree, lex_min_blue)

        return root_node if fold_successful else None

    def _fold(self, red_node, blue_node):
        # Change the output of red only to concrete output, ignore None
        stack = [(red_node, blue_node)]

        while stack:
            red, blue = stack.pop()

            # Change the output of red only to concrete output, ignore None
            if not self._compatible_states(red, blue):
                return False

            red.output = blue.output if blue.output is not None else red.output

            # Top of stack
            for k in blue.top_of_stack.keys():
                if k not in red.top_of_stack.keys():
                    red.top_of_stack[k] = blue.top_of_stack[k]

            for i in blue.children.keys():
                if i in red.children.keys():
                    stack.append((red.children[i], blue.children[i]))
                else:
                    red.children[i] = blue.children[i]

        return True
