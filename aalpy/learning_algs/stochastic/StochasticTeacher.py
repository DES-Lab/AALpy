from collections import defaultdict
from random import choice, random

from aalpy.base import SUL
from aalpy.learning_algs.stochastic.DifferenceChecker import DifferenceChecker


class StochasticSUL(SUL):
    def __init__(self, sul, teacher):
        super().__init__()
        self.sul = sul
        self.teacher = teacher

    def pre(self):
        self.num_queries += 1
        self.teacher.back_to_root()
        self.sul.pre()

    def post(self):
        self.sul.post()

    def step(self, letter):
        self.num_steps += 1
        out = self.sul.step(letter)
        self.teacher.add(letter, out)
        return out


class Node:
    """
    Node of the cache/multiset of all traces.
    """

    def __init__(self, output):
        self.output = output
        self.frequency = 0
        self.children = defaultdict(dict)
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
        if inp not in self.children.keys() or out not in self.children[inp].keys():
            return None
        return self.children[inp][out]

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
        return {child.output: child.frequency for child in self.children[input_letter].values()}


class StochasticTeacher:
    """
    The sampling-based teacher maintains a multiset of traces S for the estimation of output distributions.
    Whenever new traces are sampled in the course of learning, they are added to S.
    """

    def __init__(self, sul: SUL, n_c, eq_oracle, automaton_type, compatibility_checker: DifferenceChecker,
                 samples_cex_strategy=None):
        self.automaton_type = automaton_type
        if automaton_type == 'mdp':
            self.initial_value = sul.query(tuple())
            self.root_node = Node(self.initial_value[-1])
        else:
            self.root_node = Node(None)

        self.sul = StochasticSUL(sul=sul, teacher=self)

        self.eq_oracle = eq_oracle
        self.n_c = n_c

        self.curr_node = None
        # cache
        self.complete_query_cache = set()
        self.compatibility_checker = compatibility_checker
        self.samples_cex_strategy = samples_cex_strategy

        # eq query cache
        self.last_hyp_size = 0
        self.last_cex = None
        self.last_tree_cex = None

    def back_to_root(self):
        self.curr_node = self.root_node

    def add(self, inp, out):
        """
        Adds a input/output to the tree.

        Args:

            inp: input
            out: output


        """
        self.curr_node.input_frequencies[inp] += 1
        if inp not in self.curr_node.children.keys() or out not in self.curr_node.children[inp].keys():
            node = Node(out)
            self.curr_node.children[inp][out] = node

        self.curr_node = self.curr_node.children[inp][out]
        self.curr_node.frequency += 1

    def frequency_query(self, s: tuple, e: tuple):
        """Output frequencies observed after trace s + e.

        Args:

            s: sequence from S set
            e: sequence from E set


        Returns:

            sum of output frequencies

        """
        if self.automaton_type == 'mdp':
            s = s[1:]

        input_seq = list(s[0::2] + e[0::2])
        output_seq = list(s[1::2] + e[1::2])

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
            s = s[1:]

        input_seq = list(s[0::2] + e[0::2])
        output_seq = list(s[1::2] + e[1::2])

        # get last input
        last_input = input_seq.pop()

        curr_node = self.root_node
        for i, o in zip(input_seq, output_seq):
            new_node = curr_node.get_child(i, o)
            if not new_node:
                curr_node_complete = curr_node.get_frequency_sum(i) >= self.n_c
                # if curr_node_complete:
                #     self.complete_query_cache.add(s + e)
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
        self.sul.pre()
        curr_node = pta_root

        inputs = []
        outputs = []

        while True:

            if curr_node.children:
                frequency_sum = sum(curr_node.input_frequencies.values())
                if frequency_sum == 0:
                    # uniform sampling in case we have no information
                    inp = choice(list(curr_node.children.keys()))
                else:
                    # use float random rather than integers to be able to work with non-integer frequency information
                    selection_value = random() * frequency_sum
                    inp = None
                    for i in curr_node.input_frequencies.keys():
                        inp = i
                        selection_value -= curr_node.input_frequencies[i]
                        if selection_value <= 0:
                            break
                    # curr_node.input_frequencies[inp] -= 1

                inputs.append(inp)
                out = self.sul.step(inp)
                new_node = curr_node.get_child(inp, out)

                if new_node:
                    outputs.append(out)
                    curr_node = new_node
                else:
                    self.sul.post()
                    return
            else:
                curr_node = pta_root
                for i, o in zip(inputs, outputs):
                    self.curr_node.input_frequencies[i] -= 1
                    curr_node = curr_node.get_child(i, o)
                self.sul.post()
                return

    def single_dfs_for_cex(self, stop_prob, hypothesis):
        curr_node = self.root_node
        curr_state = hypothesis.initial_state
        if self.automaton_type == "mdp":
            trace = tuple(self.initial_value)
        else:
            trace = ()

        while True:
            rep_trace = curr_state.prefix
            if trace != rep_trace:
                for i in curr_node.children.keys():
                    freq_in_tree = self.frequency_query(trace, (i,))
                    freq_in_hyp = self.frequency_query(rep_trace, (i,))
                    if self.compatibility_checker.check_difference(freq_in_tree, freq_in_hyp):
                        return trace + (i,)
            # choose next node randomly and return None if there is no next node
            if not curr_node.children:
                return None
            i = choice(list(curr_node.children.keys()))
            if not curr_node.children[i]:
                return None
            c = choice(list(curr_node.children[i].values()))
            o = c.output
            if self.automaton_type == 'mdp':
                next_state = next(
                    (out_state[0] for out_state in curr_state.transitions[i] if out_state[0].output == o), None)
            else:
                next_state = next((out_state[0] for out_state in curr_state.transitions[i] if out_state[1] == o),
                                  None)
            if not next_state:
                return trace + (i,)
            if random() <= stop_prob:
                return None
            else:
                curr_node = c
                curr_state = next_state
                trace = trace + (i,) + (o,)

    def dfs_for_cex_in_tree(self, hypothesis, nr_traces, stop_prob):
        for i in range(nr_traces):
            cex = self.single_dfs_for_cex(stop_prob, hypothesis)
            if cex:
                return cex
        return None

    def bfs_for_cex_in_tree(self, hypothesis):
        # BFS for cex
        if self.automaton_type == "mdp":
            to_check = [(self.root_node, hypothesis.initial_state, tuple(self.initial_value))]
        else:
            to_check = [(self.root_node, hypothesis.initial_state, ())]

        while to_check:
            (curr_node, curr_state, trace) = to_check.pop(0)
            rep_trace = curr_state.prefix
            if trace != rep_trace:
                for i in curr_node.children.keys():
                    freq_in_tree = self.frequency_query(trace, (i,))
                    freq_in_hyp = self.frequency_query(rep_trace, (i,))
                    if self.compatibility_checker.check_difference(freq_in_tree, freq_in_hyp):
                        return trace + (i,)
            for i in curr_node.children.keys():
                for c in curr_node.children[i].values():
                    o = c.output
                    if self.automaton_type == 'mdp':
                        next_state = next((out_state[0] for out_state in curr_state.transitions[i]
                                           if out_state[0].output == o), None)
                    else:
                        next_state = next((out_state[0] for out_state in curr_state.transitions[i]
                                           if out_state[1] == o), None)
                    if not next_state:
                        return trace + (i,)
                    new_trace = trace + (i,) + (o,)
                    to_check.append((c, next_state, new_trace))
        return None

    def equivalence_query(self, hypothesis):
        """
        Finds and returns a counterexample

        Args:

            hypothesis: current hypothesis

        Returns:

            counterexample

        """

        if self.samples_cex_strategy:
            cex = None
            if self.samples_cex_strategy == 'bfs':
                cex = self.bfs_for_cex_in_tree(hypothesis)
            elif self.samples_cex_strategy.startswith('random'):
                split_strategy = self.samples_cex_strategy.split(":")
                try:
                    nr_traces = int(split_strategy[1])
                    stop_prob = float(split_strategy[2])
                    cex = self.dfs_for_cex_in_tree(hypothesis, nr_traces, stop_prob)
                except Exception as e:
                    print("Problem in random DFS for cex in samples:", e)
            if cex:
                self.last_tree_cex = cex
                return cex

        # Repeat same cex if it did not lead to state size increase
        if self.last_cex and len(hypothesis.states) == self.last_hyp_size:
            if random() <= 0.33:
                cex = self.eq_oracle.find_cex(hypothesis)
                if cex and len(cex) < len(self.last_cex):
                    self.last_cex = cex[:-1]
            return self.last_cex

        self.last_hyp_size = len(hypothesis.states)

        cex = self.eq_oracle.find_cex(hypothesis)
        if cex:  # remove last output
            cex = cex[:-1]

        self.last_cex = cex
        return cex
