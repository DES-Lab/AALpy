from collections import defaultdict
from random import choice, random

from aalpy.base import SUL


class Node:
    """
    Node of the cache/multiset of all traces.
    """
    def __init__(self, output):
        self.output = output
        self.frequency = 0
        self.children = defaultdict(list)
        self.input_frequencies = defaultdict(int)

    def get_child(self, inp, out):
        """

        Args:

            inp: input
            out: output

        Returns:

            Child with output that equals to `out` reached when performing `inp`. If such child does not exist,
            return None.
        """
        return next((child for child in self.children[inp] if child.output == out), None)

    def get_frequency_sum(self, input_letter):
        """
        Returns:

            number of times input was observed in current state
        """
        return self.input_frequencies[input_letter]

    def get_output_frequencies(self, input_letter):
        """
        Args:

            input_letter: input

        Returns:

            observed outputs and their frequencies for given `input_letter` in the current state

        """
        if input_letter not in self.children.keys():
            return dict()
        return {child.output: child.frequency for child in self.children[input_letter]}


class StochasticTeacher:
    """
    The sampling-based teacher maintains a multiset of traces S for the estimation of output distributions.
    Whenever new traces are sampled in the course of learning, they are added to S.
    """

    def __init__(self, sul: SUL, n_c, eq_oracle, automaton_type):
        self.sul = sul
        self.automaton_type = automaton_type
        if automaton_type == 'mdp':
            self.initial_value = self.sul.query(tuple())
            self.root_node = Node(self.initial_value[-1])
        else:
            self.root_node = Node(None)
        self.eq_oracle = eq_oracle
        self.n_c = n_c
        # cache
        self.complete_query_cache = set()

    def add(self, input_seq: tuple, output_seq):
        """
        Adds a trace and all of its prefixes to the multiset/tree of traces.

        Args:

            input_seq: inputs
            output_seq: outputs


        """
        output_seq = list(output_seq)
        if self.automaton_type == 'mdp':
            assert output_seq.pop(0) == self.root_node.output
            assert len(input_seq) == len(output_seq)

        curr_node = self.root_node
        for i, o in zip(input_seq, output_seq):
            curr_node.input_frequencies[i] += 1
            if i not in curr_node.children.keys() or o not in {child.output for child in curr_node.children[i]}:
                node = Node(o)
                curr_node.children[i].append(node)
            else:
                node = curr_node.get_child(i, o)
                assert node is not None
            curr_node = node
            curr_node.frequency += 1

    def frequency_query(self, s: tuple, e: tuple):
        """Output frequencies observed after trace s + e.

        Args:

            s: sequence from S set
            e: sequence from E set


        Returns:

            sum of output frequencies

        """
        if self.automaton_type == 'mdp':
            input_seq = list(s[1::2] + e[0::2])
            output_seq = list(s[0::2] + e[1::2])
        else:
            input_seq = list(s[0::2] + e[0::2])
            output_seq = list(s[1::2] + e[1::2])

        if self.automaton_type == 'mdp':
            assert output_seq.pop(0) == self.root_node.output
        last_input = input_seq.pop()

        curr_node = self.root_node
        for i, o in zip(input_seq, output_seq):
            curr_node = curr_node.get_child(i, o)
            if not curr_node:
                return dict()

        output_freq = curr_node.get_output_frequencies(last_input)
        if sum(output_freq.values()) >= self.n_c:
            self.complete_query_cache.add(s + e)
        return output_freq

    def complete_query(self, s: tuple, e: tuple):
        """
        Given a test sequences returns true if sufficient information is available to estimate an output distribution
        from frequency queries; returns false otherwise.

        Args:

            s: sequence from S set
            e: sequence from E set

        Returns:

            True if cell is completed, false otherwise

        """

        # extract inputs and outputs
        if s + e in self.complete_query_cache:
            return True

        if self.automaton_type == 'mdp':
            input_seq = list(s[1::2] + e[0::2])
            output_seq = list(s[0::2] + e[1::2])
        else:
            input_seq = list(s[0::2] + e[0::2])
            output_seq = list(s[1::2] + e[1::2])

        # pop first output
        if self.automaton_type == 'mdp':
            assert output_seq.pop(0) == self.root_node.output
        # get last input
        last_input = input_seq.pop()

        curr_node = self.root_node
        for i, o in zip(input_seq, output_seq):
            new_node = curr_node.get_child(i, o)
            if not new_node:
                curr_node_complete = curr_node.get_frequency_sum(i) >= self.n_c
                return curr_node_complete
            else:
                curr_node = new_node

        sum_freq = curr_node.get_frequency_sum(last_input)
        if sum_freq >= self.n_c:
            self.complete_query_cache.add(s + e)
        return sum_freq >= self.n_c

    def refine_query(self, pta_root):
        """
        Execute a refine query based on input/output trace. If at some point real outputs differ from expected
        outputs, trace to that point is added to the tree, otherwise whole trace is executed.

        Args:

            pta_root: root of the PTA

        Returns:

            number of steps taken

        """
        self.sul.num_queries += 1
        self.sul.pre()
        curr_node = pta_root
        if self.automaton_type == 'mdp':
            out = [pta_root.output]
        else:
            out = []
        executed_inputs = []

        while True:
            if curr_node.children:
                frequency_sum = sum(curr_node.input_frequencies.values())
                if frequency_sum == 0:
                    # uniform sampling in case we have no information
                    inp = choice(list(curr_node.children.keys()))
                else:
                    # use float random rather than integers to be able to work with non-integer frequency information
                    selection_value = random() * frequency_sum
                    for i in curr_node.input_frequencies.keys():
                        inp = i
                        selection_value -= curr_node.input_frequencies[i]
                        if selection_value <= 0:
                            break

                executed_inputs.append(inp)
                out.append(self.sul.step(inp))
                new_node = curr_node.get_child(inp, out[-1])
                if new_node:
                    curr_node = new_node
                else:
                    self.add(tuple(executed_inputs), out)
                    self.sul.post()
                    self.sul.num_steps += len(executed_inputs)
                    return
            else:
                self.add(tuple(executed_inputs), out)
                self.sul.post()
                self.sul.num_steps += len(executed_inputs)
                return

    def equivalence_query(self, hypothesis):
        """
        Finds and returns a counterexample

        Args:

            hypothesis: current hypothesis

        Returns:

            counterexample

        """
        cex = self.eq_oracle.find_cex(hypothesis)
        for (inputs, outputs) in self.eq_oracle.executed_traces:
            if self.automaton_type == 'mdp':
                outputs.insert(0, self.initial_value[-1])
            self.add(inputs, outputs)
        self.eq_oracle.clear_traces()
        return cex
