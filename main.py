from aalpy.utils import get_Angluin_dfa
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_Lstar
from aalpy.learning_algs.deterministic.KV import run_KV

def get_learnlib_tut_dfa():
    from aalpy.utils.AutomatonGenerators import dfa_from_state_setup

    ll_tut_dfa = {
        'q0': (False, {'a': 'q1', 'b': 'q3'}),
        'q1': (False, {'a': 'q2', 'b': 'q4'}),
        'q2': (False, {'a': 'q2', 'b': 'q5'}),
        'q3': (False, {'a': 'q4', 'b': 'q0'}),
        'q4': (False, {'a': 'q5', 'b': 'q1'}),
        'q5': (True, {'a': 'q5', 'b': 'q2'})
    }

    return dfa_from_state_setup(ll_tut_dfa)

def main():
    # Import the DFA presented in Angluin's seminal paper
    # dfa = get_Angluin_dfa()
    dfa = get_learnlib_tut_dfa()

    # Get its input alphabet
    alphabet = dfa.get_input_alphabet()

    # Create a SUL instance weapping the Anguin's automaton
    sul = DfaSUL(dfa)

    # create a random walk equivelance oracle that will perform up to 500 steps every learning round
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500, reset_after_cex=True)

    # start the L* and print the whole process in detail
    # learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa',
    #                         cache_and_non_det_check=True, cex_processing=None, print_level=3)
    #
    # # print the DOT representation of the final automaton
    # print(learned_dfa)

    # learned_dfa_ttt = run_TTT(alphabet, sul, eq_oracle, automaton_type='dfa',
    #                           cache_and_non_det_check=True, print_level=3)

    # print(learned_dfa_ttt)

    learned_dfa_kv = run_KV(alphabet, sul, eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=True, print_level=3)

    print(learned_dfa_kv)
    # learned_dfa_kv.save()

if __name__ == "__main__":
    main()
