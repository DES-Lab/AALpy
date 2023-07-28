from collections import defaultdict
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


class hW:
    def __init__(self, input_al, sul, query_for_initial_state=False):
        self.input_alphabet = input_al
        self.sul = sul
        self.query_for_initial_state = query_for_initial_state

        self.homing_sequence = []
        self.W = []

        self.global_trace = []
        self.current_homing_sequence_outputs = dict()

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
            prefix_inputs = [i for i, _ in self.global_trace[i:i + len(self.homing_sequence)]]
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
                            self.homing_sequence.extend(cont_1[0][:diff_index + 1])

                            # reset the homing sequence output dictionary
                            self.current_homing_sequence_outputs.clear()
                            self.state_map.clear()

                            # add h to W if not already present
                            prefix_already_in_W = False
                            for w in self.W:
                                if tuple(self.homing_sequence) == w[:len(self.homing_sequence)]:
                                    prefix_already_in_W = True
                                    break

                            if not prefix_already_in_W:
                                self.W.append(tuple(self.homing_sequence))

                                # clean W set of any prefix of h
                                w_to_remove = []
                                for w in self.W:
                                    if w != tuple(self.homing_sequence) and w == tuple(self.homing_sequence[:len(w)]):
                                        w_to_remove.append(w)

                                for w in w_to_remove:
                                    self.W.remove(w)

                            return False

        return True

    def check_w_ND_consistency(self):
        pass

    def step_wrapper(self, letter):
        output = self.sul.step(letter)
        self.global_trace.append((letter, output))
        self.sul.num_steps += 1
        if self.homing_sequence:
            self.check_h_ND_consistency()
        # self.check_w_ND_consistency()
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

    def process_counterexample(self, hypothesis, cex):
        pass

    def get_paths_to_reachable_states(self, model, state_map, current_state, target_states):
        paths = []
        for t in target_states:
            if current_state == t:
                paths.append(((), current_state))
                break
            path = model.get_shortest_path(state_map[current_state], state_map[t])
            if path is not None:
                paths.append((path, t))
        paths.sort(key=lambda x: len(x[0]))
        return paths

    def get_incomplete_transitions(self):
        incomplete_transitions = defaultdict(list)
        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                for w in self.W:
                    if (i, w) not in state.defined_transitions_with_w.keys():
                        incomplete_transitions[hs].append((i, w))

        return incomplete_transitions

    def create_intermediate_model(self, current_hs_response):
        mealy_state_map = dict()

        for hs, state in self.state_map.items():
            mealy_state_map[hs] = MealyState(f's{len(mealy_state_map)}')
            if tuple(self.homing_sequence) in self.current_homing_sequence_outputs[hs].keys():
                mealy_state_map[hs].prefix = self.current_homing_sequence_outputs[hs][tuple(self.homing_sequence)]

        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                w_for_input = {w: output for (ii, w), output in state.defined_transitions_with_w.items() if ii == i}
                # match to element from state definition
                if len(w_for_input.keys()) == len(self.W):
                    for _, destination_state in self.state_map.items():
                        if w_for_input == destination_state.w_values:
                            mealy_state_map[hs].transitions[i] = mealy_state_map[destination_state.hs]
                            mealy_state_map[hs].output_fun[i] = state.output_fun[i]

        mm = MealyMachine(MealyState('dummy'), list(mealy_state_map.values()))
        mm.current_state = mealy_state_map[current_hs_response]
        return mm, mealy_state_map

    def create_hypothesis(self):

        while True:
            # line 6
            hs_response = self.execute_homing_sequence()

            if hs_response not in self.current_homing_sequence_outputs.keys():
                self.current_homing_sequence_outputs[hs_response] = dict()

            print(self.current_homing_sequence_outputs)

            # if hs_response is undefined for some w
            if len(self.current_homing_sequence_outputs[hs_response].keys()) != len(self.W):
                for w in self.W:
                    if w not in self.current_homing_sequence_outputs[hs_response]:
                        w_response = self.execute_sequence(w)

                        if self.current_homing_sequence_outputs:
                            self.current_homing_sequence_outputs[hs_response][w] = w_response
                        break
            # state is defined
            else:
                if hs_response not in self.state_map:
                    self.state_map[hs_response] = ModelState(hs_response,
                                                             self.current_homing_sequence_outputs[hs_response])

                hypothesis, mealy_state_map = self.create_intermediate_model(hs_response)

                incomplete_transitions = self.get_incomplete_transitions()

                incomplete_states = [incomplete_h for incomplete_h in incomplete_transitions.keys()]

                if not incomplete_states:
                    break

                alpha, reached_state = self.get_paths_to_reachable_states(hypothesis,
                                                                          mealy_state_map,
                                                                          hs_response,
                                                                          incomplete_states)[0]

                x = incomplete_transitions[reached_state][0][0]
                w = incomplete_transitions[reached_state][0][1]

                self.execute_sequence(alpha)
                output = self.step_wrapper(x)
                w_response = self.execute_sequence(w)

                if self.state_map:
                    self.state_map[reached_state].defined_transitions_with_w[(x, w)] = w_response
                    self.state_map[reached_state].output_fun[x] = output

        return hypothesis

    def main_loop(self):

        initial_model = self.create_daisy_hypothesis()

        counter_example = self.find_counterexample(initial_model)
        last_cex_input = [counter_example[-1]]

        # set HS and W to the first counterexample
        self.homing_sequence = last_cex_input
        self.W.append(tuple(last_cex_input))

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


#model = load_automaton_from_file('DotModels/Angluin_Mealy.dot', 'mealy')
model = load_automaton_from_file('DotModels/Small_Mealy.dot', 'mealy')
#model = get_Angluin_dfa()
# print(model.compute_charaterization_set())
from random import seed

seed(3)
# model = generate_random_deterministic_automata('mealy', num_states=10, input_alphabet_size=2, output_alphabet_size=2)
# print(model)
# exit()
assert model.is_strongly_connected()

sul = MealySUL(model)
input_alphabet = model.get_input_alphabet()

learner = hW(input_alphabet, sul, query_for_initial_state=True)
learned_model = learner.main_loop()
assert bisimilar(model, learned_model)
