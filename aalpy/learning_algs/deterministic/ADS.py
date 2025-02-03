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
    def __init__(self, obs_tree, current_block):
        self.initial_node = self.construct_ads(obs_tree, current_block)
        self.current_node = self.initial_node

    def get_score(self):
        return self.initial_node.get_score()

    def construct_ads(self, obs_tree, current_block):
        # builds the ADS tree recursively by selecting optimal inputs for splitting states

        if len(current_block) == 1:
            return AdsNode.create_leaf()

        split_score = {}
        best_input = self.maximal_base_input(obs_tree.alphabet, current_block, split_score)

        partitions = self.partition_on_output(current_block, best_input)
        sub_trees = sum(len(part) for part in partitions.values())
        max_input_score = sum(self.make_subtree(obs_tree, sub_trees, part) for _, part in partitions.items())

        inputs_to_keep = [i for i, (apart, non_apart) in split_score.items() if apart + non_apart >= max_input_score]

        if not inputs_to_keep:
            raise RuntimeError("No input available during ADS computation")

        best_input = None
        best_children = None
        best_score = 0

        for input_val in inputs_to_keep:
            input_partitions = self.partition_on_output(current_block, input_val)
            sub_trees_size = sum(len(part) for part in input_partitions.values())
            input_score = 0
            children = {}

            for output, partition in input_partitions.items():
                output_score, subtree = self.compute_output_subtree(obs_tree, partition, sub_trees_size)
                input_score += output_score
                children[output] = subtree

            if input_score < max_input_score:
                continue

            if best_input is None or input_score > best_score:
                best_score = input_score
                best_input = input_val
                best_children = children

        if best_input is None:
            raise RuntimeError("Error during ADS construction")

        return AdsNode(best_input, best_children, best_score)

    def make_subtree(self, obs_tree, sub_trees, partition):
        # Constructs a subtree for a partition and calculates its score
        partition_size = len(partition)
        child_score = self.construct_ads(obs_tree, partition).get_score()
        return self.compute_reg_score(partition_size, sub_trees, child_score)

    def compute_output_subtree(self, obs_tree, partition, sub_trees):
        # Computes and scores a subtree for a specific output partition
        output_subtree = self.construct_ads(obs_tree, partition)
        output_score = self.compute_reg_score(len(partition), sub_trees, output_subtree.get_score())
        return output_score, output_subtree

    def maximal_base_input(self, alphabet, block, split_score):
        # Identifies the input with the highest ability to split the state block based on apartness
        best_input = alphabet[0]
        best_apart_pairs = 0

        for input_val in alphabet:
            partition = self.partition_on_output(block, input_val)
            non_apart_pairs = 0
            sub_trees_size = 0

            for part in partition.values():
                size = len(part)
                sub_trees_size += size
                non_apart_pairs += size * (size - 1)

            apart_pairs = sub_trees_size * (sub_trees_size - 1) - non_apart_pairs
            split_score[input_val] = (apart_pairs, non_apart_pairs)

            if apart_pairs > best_apart_pairs:
                best_apart_pairs = apart_pairs
                best_input = input_val

        return best_input

    def compute_reg_score(self, partition_size, sub_trees, child_score):
        # Calculates a score based on partition size and subtree characteristics
        return partition_size * (sub_trees - partition_size) + child_score

    def partition_on_output(self, block, input_val):
        # Partitions states in the block based on their output for a given input
        partition = defaultdict(list)

        for node in block:
            output = node.get_output(input_val)
            if output is not None:
                successor = node.get_successor(input_val)
                if successor is not None:
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

    def reset_to_root(self):
        # Resets the current ADS node to the initial root node
        self.current_node = self.initial_node
