import string
import random
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.utils import generate_random_dfa
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle, WMethodEqOracle

def compute_shortest_prefixes(dfa):
    for state in dfa.states:
        state.prefix = dfa.get_shortest_path(dfa.initial_state, state)
    return dfa

def checkConformance(alphabet, learned_dfa, states_num, sul):
    learned_dfa.characterization_set = None
    learned_dfa = compute_shortest_prefixes(learned_dfa)
    #eq_oracle = WMethodEqOracle(alphabet,sul,states_num)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 100)
    cex = eq_oracle.find_cex(learned_dfa)
    cex = None
    return True if cex == None else False


def runKV():
    alphabet_size = random.randint(0,10)
    maximum_number_states = 10
    alphabet = list(string.ascii_letters[:26])[:alphabet_size]
    num_states = random.randint(1,maximum_number_states)
    num_accepting_states = random.randint(1,num_states)

    dfa = generate_random_dfa(num_states, alphabet, num_accepting_states)

    # Get its input alphabet
    alphabet = dfa.get_input_alphabet()

    # Create a SUL instance wrapping the Angluin's automaton
    sul = DfaSUL(dfa)

    # create a random walk equivalence oracle that will perform up to 500 steps every learning round
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500, reset_after_cex=True)

    learned_dfa_kv = run_KV(alphabet, sul, eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=True, print_level=3)

    learning_result = checkConformance(alphabet, learned_dfa_kv, len(dfa.states), sul)

    return dfa if learning_result == False else None

def main():

    learned_correctly = 0
    wrongy_learned_dfas = []
    random_learning_examples = 10

    for i in range(random_learning_examples):
        res = runKV()
        if res == None:
            learned_correctly += 1
        else:
            wrongy_learned_dfas.append(res)


if __name__ == "__main__":
    main()