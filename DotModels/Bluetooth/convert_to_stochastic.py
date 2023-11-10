import random

from aalpy.SULs import AutomatonSUL
from aalpy.base import SUL
from aalpy.automata import Mdp, MdpState, StochasticMealyState, StochasticMealyMachine
from aalpy.learning_algs import run_Lstar, run_stochastic_Lstar
from aalpy.oracles import RandomWMethodEqOracle, RandomWordEqOracle
from aalpy.utils import load_automaton_from_file

model = load_automaton_from_file('CYW43455.dot', automaton_type='mealy', compute_prefixes=True)
alphabet = model.get_input_alphabet()
print(alphabet)

class ModelSUL(SUL):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_output = 'init'

    def pre(self):
        self.model.reset_to_initial()
        self.last_output = 'init'

    def post(self):
        pass

    def step(self, letter):
        if not letter:
            return self.last_output
        self.last_output = self.model.step(letter)
        return self.last_output


model_sul = ModelSUL(model)


def to_mdp():
    eq_oracle = RandomWMethodEqOracle(alphabet, model_sul)

    learned_model = run_Lstar(alphabet, model_sul, eq_oracle, 'moore')

    # CC2640R2-no-feature-req.dot
    # {'mtu_req', 'pairing_req',} have 0.3 percent chance of looping to initial state
    moore_mdp_state_map = dict()
    initial_mdp_state = None
    for state in learned_model.states:
        mdp_state = MdpState(state.state_id, state.output)
        moore_mdp_state_map[state.prefix] = mdp_state
        if not state.prefix:
            initial_mdp_state = mdp_state

    # moore_mdp_state_map['sink'] = MdpState('sink', 'NO_RESPONSE')
    assert initial_mdp_state

    for state in learned_model.states:
        mdp_state = moore_mdp_state_map[state.prefix]
        state_num = int(mdp_state.state_id[1:])

        for i in alphabet:
            reached_moore = state.transitions[i].prefix
            # if i in {'pairing_req', 'mtu_req'} and mdp_state.output != moore_mdp_state_map[reached_moore].output:
            if state_num % 2 == 0 and mdp_state.output != moore_mdp_state_map[reached_moore].output:
                mdp_state.transitions[i].append((mdp_state, 0.2))
                mdp_state.transitions[i].append((moore_mdp_state_map[reached_moore], 0.8))
            else:
                mdp_state.transitions[i].append((moore_mdp_state_map[reached_moore], 1.))

    mdp = Mdp(initial_mdp_state, list(moore_mdp_state_map.values()))
    return mdp


def to_smm():
    # CC2640R2-no-feature-req.dot
    # {'mtu_req', 'pairing_req',} have 0.3 percent chance of looping to initial state
    moore_mdp_state_map = dict()
    initial_mdp_state = None
    for state in model.states:
        mdp_state = StochasticMealyState(state.state_id)
        moore_mdp_state_map[state.prefix] = mdp_state
        if not state.prefix:
            initial_mdp_state = mdp_state

    assert initial_mdp_state

    for state in model.states:
        mdp_state = moore_mdp_state_map[state.prefix]
        state_num = int(mdp_state.state_id[1:])

        for i in alphabet:
            reached_state = state.transitions[i].prefix
            correct_output = state.output_fun[i]
            # if i in {'pairing_req', 'mtu_req'} and mdp_state.output != moore_mdp_state_map[reached_moore].output:
            if state_num % 6 == 0:
                last_out = model.compute_output_seq(model.initial_state, state.prefix[:-1]) if state.prefix else "NO_RESPONSE"
                if not last_out or last_out == correct_output:
                    last_out = 'NO_RESPONSE'
                mdp_state.transitions[i].append((mdp_state, last_out[-1], 0.2))
                mdp_state.transitions[i].append((moore_mdp_state_map[reached_state], correct_output, 0.8))
            if state_num % 5 == 0 and i in {'length_req', 'length_rsp', 'feature_rsp'} and len(state.prefix) == 2:
                mdp_state.transitions[i].append((moore_mdp_state_map[model.initial_state.prefix], 'SYSTEM_CRASH', 0.1))
                mdp_state.transitions[i].append((moore_mdp_state_map[reached_state], correct_output, 0.9))
            else:
                mdp_state.transitions[i].append((moore_mdp_state_map[reached_state], correct_output, 1.))

    mdp = StochasticMealyMachine(initial_mdp_state, list(moore_mdp_state_map.values()))
    return mdp


mdp = to_smm()
# mdp.visualize()
# exit()
# mdp.save(file_path='CC2640R2-no-feature-req-stochastic')
# exit()
# mdp.make_input_complete('self_loop')
# mdp_sul = StochasticMealySUL(mdp)
mdp_sul = AutomatonSUL(mdp.to_mdp())
eq_oracle = RandomWordEqOracle(alphabet, model_sul, num_walks=10000, min_walk_len=10, max_walk_len=100)

stochastic_model = run_stochastic_Lstar(alphabet, mdp_sul, eq_oracle, automaton_type='mdp')

# mdp = mdp.to_mdp()
# mdp.save('CYW43455_stochastic')
