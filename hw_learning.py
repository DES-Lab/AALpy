from collections import defaultdict
from random import choice

from aalpy.SULs import MealySUL
from aalpy.automata import MealyState, MealyMachine
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file, generate_random_deterministic_automata
from aalpy.utils.HelperFunctions import all_suffixes
from aalpy.utils.ModelChecking import bisimilar

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
        #self.check_w_ND_consistency()
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

    def get_paths_to_reachable_states(self, states, current_state, target_states):
        mm = MealyMachine(MealyState('dummy'), states)

        paths = []
        for t in target_states:
            path = mm.get_shortest_path(current_state, t)
            if path is not None:
                paths.append(path)
        paths.sort(key=len)
        return paths

    def get_undefined_transitions(self, states):
        undefined_transitions = list()
        for state in states:
            for i in self.input_alphabet:
                if i not in state.transitions.keys():
                    undefined_transitions.append((state, i))

        return undefined_transitions

    def get_incomplete_states_with_w(self):
        incomplete_states = dict()
        for hs_response, w_responses in self.current_homing_sequence_outputs.items():
            for w in self.W:
                if w not in w_responses.keys():
                    incomplete_states[hs_response] = w
                    break

        return incomplete_states

    @property
    def create_hypothesis(self):

        while True:

            # line 6
            hs_response = self.execute_homing_sequence()

            if hs_response not in self.current_homing_sequence_outputs.keys():
                self.current_homing_sequence_outputs[hs_response] = dict()

            print(hs_response)
            print(self.current_homing_sequence_outputs)
            # if hs_response is undefined for some w
            print('RE')

            if len(self.current_homing_sequence_outputs[hs_response].keys()) != len(self.W):
                for w in self.W:
                    if w not in self.current_homing_sequence_outputs[hs_response]:
                        w_response = self.execute_sequence(w)
                        self.current_homing_sequence_outputs[hs_response][w] = w_response
                        break
            # state if defined

            else:
                tail = self.current_homing_sequence_outputs[hs_response][tuple(self.homing_sequence)]
                self.state_map[hs_response] = MealyState(f's{len(self.state_map)}')

                # learn transitions from current state
                transition_learned = False
                for i in input_alphabet:
                    if i not in self.state_map[hs_response].transitions.keys():
                        output = self.step_wrapper(i)
                        reached_state = self.execute_homing_sequence()

                        if reached_state not in self.state_map.keys():
                            self.state_map[reached_state] = MealyState(f's{len(self.state_map)}')
                        if reached_state not in self.current_homing_sequence_outputs.keys():
                            self.current_homing_sequence_outputs[reached_state] = dict()

                        self.state_map[hs_response].transitions[i] = self.state_map[reached_state]
                        self.state_map[hs_response].output_fun[i] = output
                        transition_learned = True

                if transition_learned:
                    continue

                # if transitions form current state are learned, go to state with undefined items

                incomplete_states = self.get_incomplete_states_with_w()
                break
            #
            #     for r in self.current_homing_sequence_outputs.keys():
            #         tail = self.current_homing_sequence_outputs[r][tuple(self.homing_sequence)]
            #         # self.state_map[tail] = MealyState(f's{len(self.state_map.keys())}')
            #         self.state_map[r] = MealyState(f'{tail}')
            #
            #     undefined_transitions = self.get_undefined_transitions(self.state_map.values())
            #
            #     tail_state = self.state_map[
            #         self.current_homing_sequence_outputs[hs_response][tuple(self.homing_sequence)]]
            #
            #     reachable_states = self.get_paths_to_reachable_states(self.state_map.values(), tail_state,
            #                                                           [s[0] for s in undefined_transitions])
            #
            #     # go to state with undefined transition
            #     interruption = False
            #     if reachable_states:
            #         for i in reachable_states[0]:
            #             self.step_wrapper(i)
            #             tail_state = tail_state.transitions[i]
            #             # TODO CHECK
            #             if not self.current_homing_sequence_outputs.keys():
            #                 interruption = True
            #                 break
            #
            #     if interruption:
            #         continue
            #
            #     # lines 12 and 13
            #     transition_learned = False
            #     # learn transition
            #     for i in self.input_alphabet:
            #         if (tail_state, i) in undefined_transitions:
            #             undefined_transition_output = self.step_wrapper(i)
            #             reached_homing_sequence = self.execute_homing_sequence() # WRONG
            #
            #             if reached_homing_sequence not in self.state_map.keys():
            #                 self.state_map[reached_homing_sequence] = MealyState(f'{reached_homing_sequence}')
            #
            #             tail_state.transitions[i] = self.state_map[reached_homing_sequence]
            #             tail_state.output_fun[i] = undefined_transition_output
            #             transition_learned = True
            #             break
            #
            #     if transition_learned:
            #         continue
            #
            # if self.state_map and not self.get_undefined_transitions(self.state_map.values()) and self.get_incomplete_states() is None:
            #     break



        hypothesis = MealyMachine(MealyState('dummy'), list(self.state_map.values()))

        last_hs = tuple([o for _, o in self.global_trace[-len(self.homing_sequence):]])
        hypothesis.current_state = self.state_map[last_hs]
        print('complete', self.current_homing_sequence_outputs)
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
            hypothesis = self.create_hypothesis
            print(hypothesis)

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
            self.sul.sul.num_queries += 1

            # reset would mess up the global trace
            initial_state = tuple(self.sul.sul.query(self.homing_sequence))
            hypothesis.initial_state = self.state_map[initial_state]

        print(f'h-Learning learned {hypothesis.size} states.')
        print(f'Num Resets: {self.sul.sul.num_queries}')
        print(f'Num Steps : {self.sul.sul.num_steps}')

        return hypothesis




# model = load_automaton_from_file('DotModels/Angluin_Mealy.dot', 'mealy')
model = load_automaton_from_file('DotModels/Small_Mealy.dot', 'mealy')
# print(model.compute_charaterization_set())
from random import seed

seed(0)
# model = generate_random_deterministic_automata('mealy', num_states=10, input_alphabet_size=2, output_alphabet_size=2)
# print(model)
# exit()
assert model.is_strongly_connected()

sul = MealySUL(model)
input_alphabet = model.get_input_alphabet()

learner = hW(input_alphabet, sul, query_for_initial_state=True)
learned_model = learner.main_loop()
# learned_model.visualize()
assert bisimilar(model, learned_model)
