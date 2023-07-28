import itertools
from random import choice

from aalpy.SULs import MealySUL
from aalpy.automata import MealyState, MealyMachine
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file, generate_random_deterministic_automata
from aalpy.utils.HelperFunctions import all_suffixes, all_prefixes
from aalpy.utils.ModelChecking import bisimilar


class hLearning:
    def __init__(self, input_al, sul, query_for_initial_state=False):
        self.input_alphabet = input_al
        self.sul = sul
        self.query_for_initial_state = query_for_initial_state

        self.homing_sequence = []
        self.distinguishing_sequence = []
        self.observed_outputs = set()

        sul.pre()

    def execute_homing_sequence(self):
        observed_output = []
        for i in self.homing_sequence:
            observed_output.append(self.sul.step(i))

        self.sul.num_steps += len(self.homing_sequence)
        return tuple(observed_output)

    def execute_distinguishing_sequence(self):
        observed_output = []
        for i in self.distinguishing_sequence:
            observed_output.append(self.sul.step(i))

        self.sul.num_steps += len(self.homing_sequence)
        return tuple(observed_output)

    def create_initial_hypothesis(self):
        state = MealyState('s0')
        for i in self.input_alphabet:
            o = self.sul.step(i)
            state.transitions[i] = state
            state.output_fun[i] = o
        mm = MealyMachine(MealyState('dummy'), [state])
        mm.current_state = mm.states[0]
        return mm

    def find_counterexample(self, hypothesis, initial_round=False):

        if not initial_round:
            hypothesis.current_state = None
            current_hs = self.execute_homing_sequence()
            for state in hypothesis.states:
                if state.prefix == current_hs:
                    hypothesis.current_state = state
                    break

            assert hypothesis.current_state

        cex = []
        for _ in range(30):
            random_input = choice(self.input_alphabet)
            cex.append(random_input)
            o_sul = self.sul.step(random_input)
            o_hyp = hypothesis.step(random_input)
            self.sul.num_steps += 1

            if o_sul != o_hyp:
                return cex

        return None

    def get_paths_to_reachable_states(self, states, current_state, target_states):
        mm = MealyMachine(MealyState('dummy'), states)
        hs_to_state_map = {state.prefix: state for state in mm.states}

        paths = []
        for t in target_states:
            path = mm.get_shortest_path(hs_to_state_map[current_state], hs_to_state_map[t])
            if path:
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

    def update_hypothesis(self):
        hs_to_state_map = dict()

        current_hs = None

        while True:
            if not hs_to_state_map:
                current_hs = self.execute_homing_sequence()

                hs_to_state_map[current_hs] = MealyState(f's{len(hs_to_state_map.keys())}')
                # prefix is used to keep track of HS needed to reach a state
                hs_to_state_map[current_hs].prefix = current_hs

            undefined_transitions = self.get_undefined_transitions(hs_to_state_map.values())
            if not undefined_transitions:
                break

            transition_found = False
            # get transitions for current state
            for state, undefined_input in undefined_transitions:
                # current state has an undefined transition
                if state.prefix == current_hs:
                    transition_output = self.sul.step(undefined_input)
                    self.sul.num_steps += 1

                    transition_destination = self.execute_homing_sequence()

                    if transition_destination not in hs_to_state_map.keys():
                        hs_to_state_map[transition_destination] = MealyState(f's{len(hs_to_state_map)}')
                        hs_to_state_map[transition_destination].prefix = transition_destination

                    hs_to_state_map[current_hs].transitions[undefined_input] = hs_to_state_map[transition_destination]
                    hs_to_state_map[current_hs].output_fun[undefined_input] = transition_output

                    # print(state.prefix, undefined_input, transition_output, transition_destination)

                    #current_hs = self.execute_homing_sequence()
                    current_hs = transition_destination
                    # if current_hs not in hs_to_state_map.keys():
                    #     hs_to_state_map[current_hs] = MealyState(f's{len(hs_to_state_map)}')
                    #     hs_to_state_map[current_hs].prefix = current_hs

                    transition_found = True
                    break

            if transition_found:
                continue

            # get transitions for reachable states
            reachable_states = self.get_paths_to_reachable_states(hs_to_state_map.values(), current_hs,
                                                                  [s[0].prefix for s in undefined_transitions])

            if reachable_states:
                shortest_path = reachable_states[0]
                # reach a state with undefined transition
                traversed_path = []
                for i in shortest_path:
                    traversed_path.append(i)
                    hyp_o = hs_to_state_map[current_hs].output_fun[i]
                    sul_o = self.sul.step(i)
                    self.sul.num_steps += 1

                    if hyp_o != sul_o:
                        #self.homing_sequence[:0] = traversed_path
                        self.homing_sequence.extend(traversed_path)
                        hs_to_state_map.clear()

                        break

                    current_hs = hs_to_state_map[current_hs].transitions[i].prefix
            else:
                print('REEEEEEEEEEEEEEEEEEEEEEEEEEEE')
                assert False
                break


        hypothesis = MealyMachine(MealyState('dummy'), list(hs_to_state_map.values()))
        # hypothesis.current_state = hs_to_state_map[self.execute_homing_sequence()]

        print('size of the hypothesis', hypothesis.size)
        return hypothesis

    def main_loop(self):

        initial_model = self.create_initial_hypothesis()
        # TODO it could be that this does not include first steps asked during initial model creation
        counter_example = self.find_counterexample(initial_model, initial_round=True)
        # potential issue

        self.homing_sequence = counter_example
        # self.distinguishing_sequence = counter_example
        # self.homing_sequence.extend(counter_example)

        while True:
            hypothesis = self.update_hypothesis()
            # print(hypothesis)

            counter_example = self.find_counterexample(hypothesis)
            if counter_example is None:
                break

            print('cex', counter_example)

            # self.homing_sequence.append(counter_example[-1])

            self.homing_sequence.extend(counter_example)

            # self.homing_sequence.extend(counter_example)
            #self.distinguishing_sequence = counter_example
            # self.process_cex(hypothesis)

            # self.process_cex(hypothesis, counter_example)
            # self.homing_sequence = counter_example
            #self.homing_sequence[:0] = counter_example
            #self.homing_sequence = counter_example
            #self.homing_sequence.extend(counter_example[2:])
            # print(self.homing_sequence)
            print('Len of homing seq:', len(self.homing_sequence))
            print('Len of distin seq:', len(self.distinguishing_sequence))
            # self.homing_sequence = counter_example

        if self.query_for_initial_state:
            # reset
            self.sul.post()
            self.sul.pre()
            self.sul.num_queries += 1

            initial_state_hs = tuple(self.sul.query(self.homing_sequence))

            for state in hypothesis.states:
                if state.prefix == initial_state_hs:
                    hypothesis.initial_state = state
                    break

        print(f'h-Learning learned {hypothesis.size} states.')
        print(f'Num Resets: {self.sul.num_queries}')
        print(f'Num Steps : {self.sul.num_steps}')

        return hypothesis

    # def process_cex(self, hypothesis, counter_example):
    #     hypothesis_prefixes = [s.prefix for s in hypothesis.states]
    #     uncovered_transitions = list(itertools.product(hypothesis_prefixes, self.input_alphabet))
    #
    #     unique = set()
    #     new_mapping = dict()
    #     while uncovered_transitions:
    #         if uncovered_transitions is None:
    #             break
    #
    #         transition_found = False
    #
    #         current_state = self.execute_homing_sequence()
    #
    #         for prefix, i in uncovered_transitions:
    #             if current_state == prefix:
    #                 transition_output = self.sul.step(i) # do something with this
    #                 dist_output = self.execute_distinguishing_sequence()
    #                 new_mapping[(current_state, i)] = dist_output
    #                 unique.add((transition_output,) + dist_output)
    #                 uncovered_transitions.remove((prefix,i))
    #                 transition_found = True
    #                 break
    #
    #         if transition_found:
    #             continue
    #
    #     print('END', len(unique))


# model = load_automaton_from_file('DotModels/Angluin_Mealy.dot', 'mealy')
model = load_automaton_from_file('DotModels/hw_model.dot', 'mealy')
# print(model.compute_charaterization_set())
from random import seed

seed(5)
model = generate_random_deterministic_automata('mealy', num_states=5, input_alphabet_size=2, output_alphabet_size=2)
model.visualize()
# print(model)
# exit()
assert model.is_strongly_connected()
assert model.is_minimal()
assert model.is_strongly_connected()

sul = MealySUL(model)
input_alphabet = model.get_input_alphabet()

learner = hLearning(input_alphabet, sul, query_for_initial_state=True)
learned_model = learner.main_loop()
learned_model.visualize()

if not bisimilar(model, learned_model):
    print(model.size, learned_model.size)
    print('Models not the same')
