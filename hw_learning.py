from collections import defaultdict, deque
from random import choice

from aalpy.SULs import MealySUL
from aalpy.automata import MealyState, MealyMachine
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file, generate_random_deterministic_automata
from aalpy.utils.HelperFunctions import all_suffixes
from aalpy.utils.ModelChecking import bisimilar


class ModelState:
    def __init__(self, hs, w_values):
        self.hs = hs
        self.w_values = w_values
        self.transitions = dict()
        self.output_fun = dict()

        self.defined_transitions_with_w = defaultdict(dict)


class GraphNode:
    def __init__(self, hs):
        self.hs = hs
        self.transitions = dict()

    def find_shortest_path(self, target_state):
        if self.hs == target_state:
            return ()

        # Queue stores tuples of (current_node, path_taken)
        queue = deque([(self, ())])
        visited = set()

        while queue:
            current_node, path_taken = queue.popleft()

            if current_node.hs == target_state:
                return path_taken

            if current_node in visited:
                continue

            visited.add(current_node)

            for input_sequence, next_node in current_node.transitions.items():
                new_path = path_taken + input_sequence
                queue.append((next_node, new_path))

        return None

    def get_paths_to_reachable_states(self, target_states):
        paths = []
        for t in target_states:
            path = self.find_shortest_path(t)
            if path is not None:
                paths.append((path, t))

        paths.sort(key=lambda x: len(x[0]))
        return paths


class hW:
    def __init__(self, input_al, sul, query_for_initial_state=False):
        self.input_alphabet = input_al
        self.sul = sul
        self.query_for_initial_state = query_for_initial_state

        self.homing_sequence = ()
        self.W = []

        self.global_trace = []
        self.current_homing_sequence_outputs = dict()

        self.interrupt = False

        self.state_map = dict()
        self.transition_map = defaultdict(dict)
        self.output_map = defaultdict(dict)

        # TODO remove at the end
        sul.pre()

    def execute_homing_sequence(self):
        observed_output = []

        for i in self.homing_sequence:
            observed_output.append(self.step_wrapper(i))

        return tuple(observed_output)

    def execute_sequence(self, seq_under_test):
        observed_output = []
        for i in seq_under_test:
            observed_output.append(self.step_wrapper(i))

        observed_output = tuple(observed_output)

        if tuple(seq_under_test) == self.homing_sequence and \
                observed_output not in self.current_homing_sequence_outputs.keys():
            self.current_homing_sequence_outputs[observed_output] = dict()

        return observed_output

    def create_daisy_hypothesis(self):
        state = MealyState('s0')
        for i in self.input_alphabet:
            o = self.step_wrapper(i)
            state.transitions[i] = state
            state.output_fun[i] = o
        mm = MealyMachine(MealyState('dummy'), [state])
        mm.current_state = mm.states[0]
        return mm

    def check_h_ND_consistency(self):
        same_output_hs_continuations = defaultdict(list)
        # extract all continuations
        for i in range(len(self.global_trace)):
            prefix_inputs = tuple([i for i, _ in self.global_trace[i:i + len(self.homing_sequence)]])
            if prefix_inputs == self.homing_sequence:
                prefix_outputs = tuple([o for _, o in self.global_trace[i:i + len(self.homing_sequence)]])
                cont_inputs = tuple([i for i, _ in self.global_trace[i + len(self.homing_sequence):]])
                cont_outputs = [o for _, o in self.global_trace[i + len(self.homing_sequence):]]
                if not cont_inputs:
                    continue
                same_output_hs_continuations[prefix_outputs].append((cont_inputs, cont_outputs))

        for hs_output in same_output_hs_continuations.keys():
            if hs_output not in self.current_homing_sequence_outputs:
                self.current_homing_sequence_outputs[hs_output] = dict()

        for hs_continuations in same_output_hs_continuations.values():
            if len(hs_continuations) <= 1:
                continue
            hs_continuations.sort(key=lambda x: len(x[0]))

            for i in range(len(hs_continuations)):
                for j in range(i + 1, len(hs_continuations)):
                    cont_1, cont_2 = hs_continuations[i], hs_continuations[j]
                    for diff_index in range(len(cont_1[0])):
                        if cont_1[0][diff_index] != cont_2[0][diff_index]:
                            break
                        if cont_1[1][diff_index] != cont_2[1][diff_index]:

                            # extend the homing sequence
                            # self.homing_sequence.extend(cont_1[0][:diff_index + 1])
                            self.homing_sequence += tuple(cont_1[0][:diff_index + 1])

                            self.interrupt = True

                            # reset the homing sequence output dictionary
                            self.current_homing_sequence_outputs.clear()
                            self.state_map.clear()

                            # add h to W if not already present
                            prefix_already_in_W = False
                            for w in self.W:
                                if self.homing_sequence == tuple(w[:len(self.homing_sequence)]):
                                    prefix_already_in_W = True
                                    break

                            if not prefix_already_in_W:
                                self.W.append(self.homing_sequence)

                                # clean W set of any prefix of h
                                w_to_remove = []
                                for w in self.W:
                                    if w != self.homing_sequence and w == tuple(self.homing_sequence[:len(w)]):
                                        w_to_remove.append(w)

                                for w in w_to_remove:
                                    self.W.remove(w)

                            return False

        return True

    def check_w_ND_consistency(self, connectivity_graph):
        same_states = []
        graph_nodes = list(connectivity_graph.items())
        state_continuations = defaultdict(list)

        for ind, (h, graph_node) in enumerate(graph_nodes):
            for h_prime, graph_node_prime in graph_nodes[ind + 1:]:
                if graph_node.transitions and graph_node.transitions == graph_node_prime.transitions:
                    s1, s2 = tuple(zip(self.homing_sequence, h)), tuple(zip(self.homing_sequence, h_prime))
                    same_states.append((s1, s2))

                    # extract continuation sequances for states
                    for s in [s1, s2]:
                        for i in range(len(self.global_trace)):
                            if tuple(self.global_trace[i:i + len(s)]) == s:
                                state_continuations[s].append(self.global_trace[i + len(s):])

        for s1, s2 in same_states:
            for s1_cont in state_continuations[s1]:
                for s2_cont in state_continuations[s2]:
                    disagreement_index = -1
                    for i in range(min(len(s1_cont), len(s2_cont))):
                        # if inputs are not the same break
                        if s1_cont[i][0] != s2_cont[i][0]:
                            break
                        if s1_cont[i] != s2_cont[i]:
                            disagreement_index = i
                            break

                    if disagreement_index >= 0:
                        print(s1_cont[disagreement_index:])
                        print(s2_cont[disagreement_index:])
                        print('REEE')
                        exit(523)

    def step_wrapper(self, letter):
        output = self.sul.step(letter)
        self.global_trace.append((letter, output))
        self.sul.num_steps += 1
        if self.homing_sequence:
            self.check_h_ND_consistency()
        return output

    def find_counterexample(self, hypothesis):

        cex = []
        for _ in range(40):
            random_input = choice(self.input_alphabet)
            cex.append(random_input)
            o_sul = self.step_wrapper(random_input)
            o_hyp = hypothesis.step(random_input)

            if o_sul != o_hyp:
                return cex

        return None

    def all_states_defined(self):
        for observed_w_seq in self.current_homing_sequence_outputs.values():
            if len(observed_w_seq.keys()) != len(self.W):
                return False
        return True

    def get_incomplete_transitions(self):
        incomplete_transitions = defaultdict(list)
        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                for w in self.W:
                    if (i, w) not in state.defined_transitions_with_w.keys():
                        incomplete_transitions[hs].append((i, w))

        for hs, observed_w_seq in self.current_homing_sequence_outputs.items():
            for w in self.W:
                if w not in observed_w_seq.keys():
                    incomplete_transitions[hs].append((None, w))

        return incomplete_transitions

    def create_connectivity_graph(self):
        nodes = dict()

        # include also undefined nodes
        for hs, state in self.current_homing_sequence_outputs.items():

            nodes[hs] = GraphNode(hs)

        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                w_for_input = {w: output for (ii, w), output in state.defined_transitions_with_w.items() if ii == i}
                # match to element from state definition
                if len(w_for_input.keys()) == len(self.W):
                    for _, destination_state in self.state_map.items():
                        if w_for_input == destination_state.w_values:
                            nodes[hs].transitions[tuple([i, ])] = nodes[destination_state.hs]

            for (x, w), destination in state.defined_transitions_with_w.items():
                if w == self.homing_sequence:
                    # why is this needed, angluin with seed 3 breaks it
                    if destination not in nodes and tuple([x, ]) not in nodes[hs].transitions.keys():
                        continue
                    if self.homing_sequence not in self.current_homing_sequence_outputs[destination]:
                        continue
                    nodes[hs].transitions[tuple([x, ])] = nodes[destination]

        return nodes

    def get_tail(self, x):
        return self.current_homing_sequence_outputs[x][self.homing_sequence]

    def create_model(self, current_hs):
        automata_states = dict()

        # include also undefined nodes
        for hs, state in self.current_homing_sequence_outputs.items():
            tail = self.get_tail(hs)

            automata_states[tail] = MealyState(f's{len(automata_states)}')
            if self.homing_sequence in self.current_homing_sequence_outputs[hs].keys():
                automata_states[tail].prefix = tail

        for hs, state in self.state_map.items():
            tail = self.get_tail(hs)

            for i in self.input_alphabet:
                w_for_input = {w: output for (ii, w), output in state.defined_transitions_with_w.items() if ii == i}
                # match to element from state definition
                print(f'Origin: {tail}, {i}, Target: {w_for_input}')
                if len(w_for_input.keys()) == len(self.W):
                    for _, destination_state in self.state_map.items():
                        if w_for_input == destination_state.w_values:
                            automata_states[tail].transitions[i] = automata_states[self.get_tail(destination_state.hs)]
                            automata_states[tail].output_fun[i] = state.output_fun[i]
                            break

        mm = MealyMachine(MealyState('dummy'), list(automata_states.values()))
        mm.current_state = automata_states[self.get_tail(current_hs)]

        return mm

    def create_hypothesis(self):

        while True:
            # line 6
            hs_response = self.execute_homing_sequence()

            # Check if querying a homing sequence yielded an interrupt
            if self.interrupt:
                self.interrupt = False
                continue

            if hs_response not in self.current_homing_sequence_outputs.keys():
                self.current_homing_sequence_outputs[hs_response] = dict()

            for state in self.current_homing_sequence_outputs.items():
                print(state)

            # if hs_response is undefined for some w
            if len(self.current_homing_sequence_outputs[hs_response].keys()) != len(self.W):
                for w in self.W:
                    if w not in self.current_homing_sequence_outputs[hs_response]:
                        w_response = self.execute_sequence(w)

                        if not self.interrupt:
                            self.current_homing_sequence_outputs[hs_response][w] = w_response
                        break

            # state is defined
            else:
                if hs_response not in self.state_map:
                    self.state_map[hs_response] = ModelState(hs_response,
                                                             self.current_homing_sequence_outputs[hs_response])

                # why here, ugly
                for hs, defined_w in self.current_homing_sequence_outputs.items():
                    if len(defined_w.keys()) == len(self.W) and hs not in self.state_map.keys():
                        self.state_map[hs] = ModelState(hs, defined_w)

                connectivity_graph = self.create_connectivity_graph()
                self.check_w_ND_consistency(connectivity_graph)

                current_node = connectivity_graph[hs_response]

                incomplete_transitions = self.get_incomplete_transitions()

                incomplete_states = [incomplete_h for incomplete_h in incomplete_transitions.keys()]

                if not incomplete_states:
                    break

                alpha, reached_state = current_node.get_paths_to_reachable_states(incomplete_states)[0]

                x = incomplete_transitions[reached_state][0][0]
                w = incomplete_transitions[reached_state][0][1]

                # execute sequence to reach a state
                self.execute_sequence(alpha)

                output = None

                if x is not None:
                    output = self.step_wrapper(x)
                w_response = self.execute_sequence(w)

                if self.interrupt:
                    self.interrupt = False
                    continue

                if self.state_map and x is not None:
                    self.state_map[reached_state].defined_transitions_with_w[(x, w)] = w_response
                    self.state_map[reached_state].output_fun[x] = output
                else:
                    if self.current_homing_sequence_outputs:
                        self.current_homing_sequence_outputs[reached_state][w] = w_response

        print('State definitions')
        print(self.current_homing_sequence_outputs)

        hypothesis = self.create_model(hs_response)

        return hypothesis

    def main_loop(self):

        initial_model = self.create_daisy_hypothesis()

        counter_example = self.find_counterexample(initial_model)
        last_cex_input = tuple([counter_example[-1]])

        # set HS and W to the first counterexample
        self.homing_sequence = last_cex_input
        self.W.append(last_cex_input)

        # add reference of homing sequance to SUL wrapper
        self.sul.h = self.homing_sequence

        while True:
            hypothesis = self.create_hypothesis()
            counter_example = self.find_counterexample(hypothesis)

            if counter_example is None:
                break

            cex_suffixes = all_suffixes(counter_example)
            suffix_added = False

            for s in cex_suffixes:
                for w in self.W:
                    if s == w[-len(s):] or s in self.W:
                        continue

                    self.W.append(s)
                    suffix_added = True
                    break
                if suffix_added:
                    break

            print(self.W)
            self.state_map.clear()

        if self.query_for_initial_state:
            # reset
            self.sul.pre()
            self.sul.num_queries += 1

            # call query as reset would mess up the global trace
            initial_state_hs = tuple(self.sul.query(self.homing_sequence))
            for s in hypothesis.states:
                if s.prefix == initial_state_hs:
                    hypothesis.initial_state = s
                    break

        print(f'h-Learning learned {hypothesis.size} states.')
        print(f'Num Resets: {self.sul.num_queries}')
        print(f'Num Steps : {self.sul.num_steps}')

        return hypothesis


model = load_automaton_from_file('DotModels/Angluin_Mealy.dot', 'mealy')
# model = load_automaton_from_file('DotModels/hw_model.dot', 'mealy')
# model = load_automaton_from_file('DotModels/Small_Mealy.dot', 'mealy')
# model = get_Angluin_dfa()
# print(model.compute_charaterization_set())
from random import seed

seed(3)
# model = generate_random_deterministic_automata('mealy', num_states=20, input_alphabet_size=2, output_alphabet_size=3)
# print(model)
# exit()
assert model.is_strongly_connected()

sul = MealySUL(model)
input_alphabet = model.get_input_alphabet()

learner = hW(input_alphabet, sul, query_for_initial_state=True)
learned_model = learner.main_loop()
assert learned_model.is_minimal()
assert bisimilar(model, learned_model)
