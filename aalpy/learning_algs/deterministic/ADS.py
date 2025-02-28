from collections import defaultdict


class AdsNode:
    __slots__ = ['input', 'children', 'score']

    def __init__(self, input_val=None, children=None, score=0):
        self.input = input_val
        self.children = children if children else {}
        self.score = score

    @staticmethod
    def create_leaf():
        return AdsNode()

    def get_input(self):
        return self.input

    def get_child_node(self, output):
        if output in self.children:
            return self.children[output]
        return None

    def get_score(self):
        return self.score


class Ads:
    def __init__(self, ob_tree, current_block):
        self.initial_node = self.construct_ads(ob_tree, current_block)
        self.current_node = self.initial_node

    def get_score(self):
        return self.initial_node.get_score()

    def construct_ads(self, ob_tree, current_block):
        # Builds the ADS tree recursively by selecting optimal inputs for splitting states
        # For DFA/Moore we have to consider the output for the empty word
        if ob_tree.automaton_type == 'mealy':
            return self.construct_ads_rec(ob_tree, current_block)
        else:
            if len(current_block) == 1:
                return AdsNode.create_leaf()

            # if none of the nodes in the current block have a successor, we cannot decide a next input
            if not any([True for node in current_block if node.successors is not None]):
                raise RuntimeError("No input available during ADS computation")

            input = tuple()
            empty_part = self.partition_on_output_empty(current_block, ob_tree.automaton_type)
            u_i = sum(len(part) for part in empty_part.values())
            score = 0
            children = {}

            for output, partition in empty_part.items():
                output_score, subtree = self.compute_output_subtree(ob_tree, partition, u_i) 
                score += output_score
                children[output] = subtree

            return AdsNode(input, children, score)

    def construct_ads_rec(self, ob_tree, current_block):
        # Builds the ADS tree recursively by selecting optimal inputs for splitting states
        if len(current_block) == 1:
            return AdsNode.create_leaf()

        # If none of the nodes in the current block have a successor, we cannot decide a next input
        if not any([True for node in current_block if node.successors is not None]):
            raise RuntimeError("No input available during ADS computation")

        best_input, best_score = self.maximal_base_input(ob_tree.alphabet, current_block, ob_tree.automaton_type)
        best_children = None

        # The maximal apartness pairs is len(current block) - 1, for any current block, immediately return
        if best_score == len(current_block) - 1:
            return AdsNode(best_input, best_children, best_score)

        for input_val in ob_tree.alphabet:
            input_partitions = self.partition_on_output(current_block, input_val, ob_tree.automaton_type)
            u_i = sum(len(part) for part in input_partitions.values())
            input_score = 0
            children = {}

            for output, partition in input_partitions.items():
                output_score, subtree = self.compute_output_subtree(ob_tree, partition, u_i) 
                input_score += output_score
                children[output] = subtree

            if input_score > best_score:
                best_score = input_score
                best_input = input_val
                best_children = children
            if best_score == len(current_block) - 1:
                return AdsNode(best_input, best_children, best_score)

        if best_input is None:
            raise RuntimeError("Error during ADS construction")

        return AdsNode(best_input, best_children, best_score)

    # def make_subtree(self, ob_tree, sub_trees, partition):
    #     # Constructs a subtree for a partition and calculates its score
    #     partition_size = len(partition)
    #     child_score = self.construct_ads_rec(ob_tree, partition).get_score()
    #     return self.compute_reg_score(partition_size, sub_trees, child_score)

    def compute_output_subtree(self, ob_tree, partition, u_i):
        # Computes and scores a subtree for a specific output partition
        output_subtree = self.construct_ads_rec(ob_tree, partition)
        output_score = self.compute_score(len(partition), u_i, output_subtree.get_score())
        return output_score, output_subtree

    def compute_score(self, u_io, u_i, child_score):
        # Calculates a score based on partition size and subtree characteristics
        return (u_io * (u_i - u_io + child_score)) / u_i

    def partition_on_output_empty(self, block, automaton_type):
        # Partitions states in the block based on their output for the empty word
        # Only use during the initial call
        partition = defaultdict(list)
        for node in block:
            output = node.output
            partition[output].append(node)
        return partition

    def partition_on_output(self, block, input_val, automaton_type):
        # Partitions states in the block based on their output for a given input
        partition = defaultdict(list)
        for node in block:
            if automaton_type == 'mealy':
                output = node.get_output(input_val)
                if output is not None:
                    successor = node.get_successor(input_val)
                    if successor is not None:
                        partition[output].append(successor)
            else:
                successor = node.get_successor(input_val)
                if successor is not None:
                    output = successor.output 
                    if output is not None:
                        partition[output].append(successor)
        return partition

    def next_input(self, prev_output):
        # Returns the next input based on the previous output and updates the current node
        if prev_output is not None:
            child = self.current_node.get_child_node(prev_output)
            if child is None:
                return None
            self.current_node = child
        return self.current_node.get_input()

    def maximal_base_input(self, alphabet, block, automaton_type):
        # Identifies the input with the highest ability to split the state block based on apartness
        # Does not use the recursive part of the formula
        best_input = alphabet[0]
        best_score = 0

        for input_val in alphabet:
            partition = self.partition_on_output(block, input_val, automaton_type)
            u_i = sum(len(part) for part in partition.values())
            score = 0

            for part in partition.values():
                u_io = len(part)
                output_score = (u_io * (u_i - u_io)) / u_i
                score += output_score

            if score > best_score:
                best_score = score
                best_input = input_val

        return best_input, best_score

    def reset_to_root(self):
        # Resets the current ADS node to the initial root node
        self.current_node = self.initial_node