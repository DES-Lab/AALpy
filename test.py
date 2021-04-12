# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from random import seed

# By defining the radnom seed, experiment is repr

seed(5)


# %%
from aalpy.utils import generate_random_mealy_machine

# define parameters and create random Mealy machine based on them

number_of_states = 400
alphabet_size = 100
output_size = 100


alphabet = [*range(0, alphabet_size)]

random_mealy = generate_random_mealy_machine(number_of_states, alphabet, output_alphabet=list(range(output_size)))


# %%
from aalpy.SULs import MealySUL

# wrap the randomly generated Mealy machine in SUL

sul_mealy = MealySUL(random_mealy)


# %%
from aalpy.oracles import StatePrefixEqOracle, KWayTransitionCoverageEqOracle, RandomWMethodEqOracle

# create the equivelance oracle

state_origin_eq_oracle = StatePrefixEqOracle(alphabet, sul_mealy, walks_per_state=10, walk_len=15)

random_W_method_eq_oracle = RandomWMethodEqOracle(alphabet, sul_mealy, walks_per_state=10, walk_len=50)

k_way_transition_coverage_eq_oracle = KWayTransitionCoverageEqOracle(alphabet, sul_mealy, k=2, random_walk_len=0)


# %%
from aalpy.learning_algs import run_Lstar

# start learning with Shabaz-Groz counter-example processing

learned_mealy = run_Lstar(alphabet, sul_mealy, random_W_method_eq_oracle, automaton_type='mealy')

# print(learned_mealy)

print('################################')

learned_mealy = run_Lstar(alphabet, sul_mealy, k_way_transition_coverage_eq_oracle, automaton_type='mealy')

# %%
# print the DOT represetnation of the learned Mealy machine
# print(learned_mealy)


