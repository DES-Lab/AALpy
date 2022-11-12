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
    #eq_oracle = WMethodEqOracle(alphabet,sul,states_num+1)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 10000)
    cex = eq_oracle.find_cex(learned_dfa)
    cex = None
    return True if cex == None else False


def runKV(seed):
    print(f"using {seed=}")
    random.seed(seed)
    alphabet_size = random.randint(1,20)
    maximum_number_states = 10
    alphabet = list(string.ascii_letters[:26])[:alphabet_size]
    num_states = random.randint(1,maximum_number_states)
    num_accepting_states = random.randint(1,num_states)

    dfa = generate_random_dfa(num_states, alphabet, num_accepting_states,ensure_minimality=False)

    # Get its input alphabet
    alphabet = dfa.get_input_alphabet()

    # Create a SUL instance wrapping the random automaton
    sul = DfaSUL(dfa)

    # create a random walk equivalence oracle that will perform up to 500 steps every learning round
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500, reset_after_cex=True)

    learned_dfa_kv = run_KV(alphabet, sul, eq_oracle, automaton_type='dfa',
                            print_level=3, reuse_counterexamples=True, cex_processing='rs')

    learning_result = checkConformance(alphabet, learned_dfa_kv, len(dfa.states), sul)

    return dfa if learning_result == False else None

def main():

    learned_correctly = 0
    wrongy_learned_dfas = []
    random_learning_examples = 10

    for i in range(random_learning_examples):
        res = runKV(random.randint(0,2000))
        if res == None:
            learned_correctly += 1
        else:
            print(f"seed for wrongly learned DFA: {i}")
            wrongy_learned_dfas.append(i)
    
    print(f'learned correctly: {learned_correctly}/{random_learning_examples}')


if __name__ == "__main__":
    main()