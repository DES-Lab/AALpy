from collections import defaultdict, deque
from random import choice

from aalpy.SULs import MealySUL
from aalpy.automata import MealyState, MealyMachine
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file, generate_random_deterministic_automata
from aalpy.utils.HelperFunctions import all_suffixes
from aalpy.utils.ModelChecking import bisimilar


class GraphNode:
    def __init__(self, hs):
        self.hs = hs
        self.tail = None

        self.transitions = dict()
        self.output_fun = dict()

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

            for i, next_node in current_node.transitions.items():
                path_taken += tuple([i, ])
                queue.append((next_node, path_taken))

        return None

    def get_paths_to_reachable_states(self, target_states, homing_seq):
        paths = []
        for t in target_states:
            path = self.find_shortest_path(t)
            if path is not None:
                paths.append((path, t))

        # paths.append((homing_seq, self.tail.hs))

        paths.sort(key=lambda x: len(x[0]))

        return paths

    def execute_seq(self, x):
        output = ()
        current_state = self
        for i in x:
            output += tuple([current_state.output_fun[i], ])
            current_state = current_state.transitions[i]

        return output


class eW:
    def __init__(self, input_al, sul, query_for_initial_state=False):
        self.input_alphabet = input_al
        self.sul = sul
        self.query_for_initial_state = query_for_initial_state

        self.homing_sequence = ()

        self.global_trace = []

        self.interrupt = False

        self.state_map = dict()

        # TODO remove at the end
        sul.pre()

    def execute_homing_sequence(self):
        observed_output = []

        for i in self.homing_sequence:
            observed_output.append(self.step_wrapper(i))

        observed_output = tuple(observed_output)

        if not self.interrupt and observed_output not in self.state_map.keys():
            self.state_map[observed_output] = GraphNode(observed_output)

        return tuple(observed_output)

    def execute_sequence(self, seq_under_test):
        observed_output = []
        for i in seq_under_test:
            observed_output.append(self.step_wrapper(i))

        return tuple(observed_output)

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
            if hs_output not in self.state_map.keys():
                self.state_map[hs_output] = GraphNode(hs_output)

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
                            self.state_map.clear()
                            print(len(self.homing_sequence))

                            return False

        return True

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

    def get_undefined_transitions(self):

        undefined_transitions = defaultdict(list)
        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                if i not in state.transitions.keys():
                    undefined_transitions[hs].append(i)

        return undefined_transitions

    def construct_hypothesis(self, current_hs):
        current_hs = self.execute_homing_sequence()
        automaton_states = dict()

        for hs, state in self.state_map.items():
            automaton_states[hs] = MealyState(f's{len(automaton_states)}')
            automaton_states[hs].prefix = state.hs

        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                automaton_states[hs].transitions[i] = automaton_states[state.transitions[i].hs]
                automaton_states[hs].output_fun[i] = state.output_fun[i]

        mm = MealyMachine(MealyState('Dummy Initial'), list(automaton_states.values()))
        tail = self.state_map[current_hs].tail.hs
        mm.current_state = automaton_states[tail]

        return mm

    def get_tail_state(self, x):
        for state in self.state_map.values():
            if state.hs == x:
                return state.tail
        return None

    def learn(self):

        while True:

            hs_response = self.execute_homing_sequence()

            # Check if querying a homing sequence yielded an interrupt
            if self.interrupt:
                self.interrupt = False
                continue

            current_state = self.state_map[hs_response]

            if current_state.tail is None:
                tail = self.execute_homing_sequence()
                if self.interrupt:
                    self.interrupt = False
                    continue
                current_state.tail = self.state_map[tail]
                continue

            #################
            current_state = current_state.tail

            undefined_transitions = self.get_undefined_transitions()

            if not undefined_transitions:
                break

            paths_to_undefined = current_state.get_paths_to_reachable_states(list(undefined_transitions.keys()),
                                                                             self.homing_sequence)

            paths_to_undefined = paths_to_undefined[0]

            alpha, target_state = paths_to_undefined[0], paths_to_undefined[1]
            undefined_input = undefined_transitions[target_state][0]

            transition_output = self.execute_sequence(alpha)
            model_output = current_state.execute_seq(alpha)

            if transition_output != model_output:
                print('Transmission error')
                continue
                # continue

            if self.interrupt:
                print('Interrupt after transition')
                self.interrupt = False
                continue

            # IMPORTANT, MAYBE ALSO IN HW
            tail_state = self.state_map[target_state]

            # transition
            output = self.step_wrapper(undefined_input)
            # homing sequance
            reached_state = self.execute_homing_sequence()

            if self.interrupt:
                self.interrupt = False
                continue

            tail_state.output_fun[undefined_input] = output
            tail_state.transitions[undefined_input] = self.state_map[reached_state]

        hypothesis = self.construct_hypothesis(hs_response)
        return hypothesis

    def main_loop(self):

        initial_model = self.create_daisy_hypothesis()

        counter_example = self.find_counterexample(initial_model)
        last_cex_input = tuple([counter_example[-1]])

        # set HS and W to the first counterexample
        self.homing_sequence = last_cex_input

        # add reference of homing sequance to SUL wrapper
        self.sul.h = self.homing_sequence

        while True:
            hypothesis = self.learn()
            counter_example = self.find_counterexample(hypothesis)

            if counter_example is None:
                break

            self.state_map.clear()


            # print('CEX', counter_example)
            # print('HS', self.homing_sequence)
            # for s in all_suffixes(counter_example):
            #     if self.homing_sequence[-len(s):] != s:
            #         print('added suf', s)
            #         self.homing_sequence += tuple(s)
            #         print('NEW HS', self.homing_sequence)
            #         break

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

seed(2)
model = generate_random_deterministic_automata('mealy', num_states=4, input_alphabet_size=2, output_alphabet_size=2)
#model.visualize()
#exit()
# print(model)
# exit()
assert model.is_strongly_connected()

sul = MealySUL(model)
input_alphabet = model.get_input_alphabet()

learner = eW(input_alphabet, sul, query_for_initial_state=True)
learned_model = learner.main_loop()

# learned_model.visualize()
assert learned_model.is_minimal()
assert bisimilar(model, learned_model)
