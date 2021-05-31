import random

from aalpy.automata import Dfa, DfaState, MdpState, Mdp, MealyMachine, MealyState, \
    MooreMachine, MooreState, OnfsmState, Onfsm
from aalpy.utils.HelperFunctions import random_string_generator

#
# s1 = State('s1')
# s2 = State('s2')
# s3 = State('s3')
# s4 = State('s4')
# s5 = State('s5')
# s6 = State('s6')
#
# # a&b - a
# # a&!b - b
# # !a&b - c
# # !a&!b - d
#
# # s1 out
# s1.add_edge('a', s2, 0.2)
# s1.add_edge('b', s1, 0.35)
# s1.add_edge('d', s3, 0.35)
# s1.add_edge('c', s1, 0.1)
#
# # s2 out
# s2.add_edge('a', s4, 0.4)
# s2.add_edge('b', s5, 0.1)
# s2.add_edge('c', s6, 0.35)
# s2.add_edge('d', s1, 0.15)
#
# # s3 out
# s3.add_edge('a', s6, 0.4)
# s3.add_edge('b', s1, 0.2)
# s3.add_edge('d', s2, 0.2)
# s3.add_edge('c', s3, 0.2)
#
# # s4 out
# s4.add_edge('a', s5, 0.25)
# s4.add_edge('c', s1, 0.25)
# s4.add_edge('b', s3, 0.5)
# s4.add_edge('d', s4, 0)
#
# # s5 out
# s5.add_edge('b', s6, 0.2)
# s5.add_edge('d', s3, 0.8)
# s5.add_edge('c', s5, 0)
# s5.add_edge('a', s5, 0)
#
# s6.add_edge('a', s6, 0.1)
# s6.add_edge('b', s6, 0.4)
# s6.add_edge('c', s6, 0.25)
# s6.add_edge('d', s6, 0.25)
#
# sm = DFA(s1, [s1, s2, s3, s4, s5, s6])

def generate_bb_mdp():
    states = []

    s1 = MdpState('s1', 's1')
    s2 = MdpState('s2', 's2')
    s3 = MdpState('s3', 's3')
    s4 = MdpState('s4', 's4')
    s5 = MdpState('s5', 's5')
    s6 = MdpState('s6', 's6')

    s1.transitions['a'].append((s2, 0.2))
    s1.transitions['b'].append((s1, 0.35))
    s1.transitions['c'].append((s3, 0.35))
    s1.transitions['d'].append((s1, 0.1))

    s2.transitions['a'].append((s4, 0.4))
    s2.transitions['b'].append((s5, 0.1))
    s2.transitions['c'].append((s6, 0.35))
    s2.transitions['d'].append((s1, 0.15))

    s3.transitions['a'].append((s6, 0.4))
    s3.transitions['b'].append((s1, 0.2))
    s3.transitions['c'].append((s3, 0.2))
    s3.transitions['d'].append((s2, 0.2))

    s4.transitions['a'].append((s5, 0.25))
    s4.transitions['b'].append((s3, 0.5))
    s4.transitions['c'].append((s1, 0.25))
    s4.transitions['d'].append((s4, 0))

    s5.transitions['a'].append((s5, 0))
    s5.transitions['b'].append((s6, 0.2))
    s5.transitions['c'].append((s5, 0))
    s5.transitions['d'].append((s3, 0.8))

    s6.transitions['a'].append((s6, 0.1))
    s6.transitions['b'].append((s6, 0.4))
    s6.transitions['c'].append((s6, 0.25))
    s6.transitions['d'].append((s6, 0.25))

    states.append(s1)
    states.append(s2)
    states.append(s3)
    states.append(s4)
    states.append(s5)
    states.append(s6)


    return Mdp(states[0], states), ["a", "b", "c", "d"]


from aalpy.SULs import MdpSUL
from aalpy.oracles import UnseenOutputRandomWalkEqOracle
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.utils import generate_random_mdp, visualize_automaton

n_c=20
n_resample=1000
min_rounds=10
max_rounds=1000

mdp, input_alphabet = generate_bb_mdp()
visualize_automaton(mdp, path="graphs/original")
sul = MdpSUL(mdp)
eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=5000, reset_prob=0.11,
                                           reset_after_cex=True)

learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=n_c, n_resample=n_resample,
                                   min_rounds=min_rounds, max_rounds=max_rounds)

visualize_automaton(learned_mdp, path="graphs/learned")
