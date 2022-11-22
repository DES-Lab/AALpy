import string
import random
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.utils import generate_random_mealy_machine
from aalpy.SULs import MealySUL
from aalpy.oracles import RandomWalkEqOracle, WMethodEqOracle

def compute_shortest_prefixes(automaton):
    for state in automaton.states:
        state.prefix = automaton.get_shortest_path(automaton.initial_state, state)
    return automaton

def checkConformance(alphabet, learned_mealy, states_num, sul):
    learned_mealy.characterization_set = None
    learned_mealy = compute_shortest_prefixes(learned_mealy)
    #eq_oracle = WMethodEqOracle(alphabet,sul,states_num+1)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 10000)
    cex = eq_oracle.find_cex(learned_mealy)
    cex = None
    return True if cex == None else False


def runKV(seed):
    print(f"using {seed=}")
    random.seed(seed)
    input_alphabet_size = random.randint(1,20)
    output_alphabet_size = random.randint(1,20)
    input_alphabet = list(string.ascii_letters[:26])[:input_alphabet_size]
    output_alphabet = list(string.ascii_letters[:26])[:output_alphabet_size]
    maximum_number_states = 100
    num_states = random.randint(1,maximum_number_states)

    mealy = generate_random_mealy_machine(num_states, input_alphabet, output_alphabet)
    print("mealy generated")
    # Get its input alphabet
    alphabet = mealy.get_input_alphabet()

    # Create a SUL instance wrapping the random automaton
    sul = MealySUL(mealy)

    # create a random walk equivalence oracle that will perform up to 500 steps every learning round
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500, reset_after_cex=True)

    learned_dfa_kv = run_KV(alphabet, sul, eq_oracle, automaton_type='mealy',
                            print_level=3, cex_processing=None)

    learning_result = checkConformance(alphabet, learned_dfa_kv, len(mealy.states), sul)

    return mealy if learning_result == False else None

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