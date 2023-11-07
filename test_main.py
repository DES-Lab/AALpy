import ast
import random

from Examples import learning_context_free_grammar_example
from aalpy.SULs.AutomataSUL import SevpaSUL, VpaSUL
from aalpy.base import SUL
from aalpy.learning_algs import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle, StatePrefixEqOracle
from aalpy.utils import visualize_automaton, get_Angluin_dfa, generate_random_sevpa
from aalpy.utils.BenchmarkVpaModels import *
from aalpy.utils.BenchmarkSevpaModels import *
from random import seed


# learning_context_free_grammar_example()

# TODOs
# 1. exponential cex processing in CounterExampleProcessing.py
# 2. Create a SEVPA function that generates random positive strings - model.generate_random_positive_string()
# 2. Add all 15 langs as SVEPA
# 4. Implement and test to_state_setup, test saving and loading to/from file
# 5. Create an active interface to learn a grammar of some language, like simplified C or Java

def test_arithmetic_expression():
    import warnings
    warnings.filterwarnings("ignore")

    class ArithmeticSUL(SUL):
        def __init__(self):
            super().__init__()
            self.string_under_test = ''

        def pre(self):
            self.string_under_test = ''

        def post(self):
            pass

        def step(self, letter):
            if letter:
                self.string_under_test += ' ' + letter

            try:
                eval(self.string_under_test)
                return True
            except (SyntaxError, TypeError):
                return False

    sul = ArithmeticSUL()
    alphabet = SevpaAlphabet(internal_alphabet=['1', '+'], call_alphabet=['('], return_alphabet=[')'])
    eq_oracle = RandomWordEqOracle(alphabet.get_merged_alphabet(), sul, min_walk_len=5,
                                   max_walk_len=20, num_walks=20000)

    learned_model = run_KV(alphabet, sul, eq_oracle, automaton_type='vpa')
    learned_model.visualize()
    exit()


def test_on_random_svepa():
    random_svepa = generate_random_sevpa(num_states=50, internal_alphabet_size=3,
                                         call_alphabet_size=3,
                                         return_alphabet_size=3,
                                         acceptance_prob=0.4,
                                         return_transition_prob=0.5)

    alphabet = random_svepa.input_alphabet

    sul = SevpaSUL(random_svepa, include_top=False, check_balance=False)

    eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000,
                                   min_walk_len=10, max_walk_len=30)
    # model = run_KV_vpda(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3,)
    model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                   print_level=2, cex_processing='rs')


def test_random_word_gen():
    model_under_learning = vpa_for_L11()

    # Learn Model
    alphabet = SevpaAlphabet(list(model_under_learning.internal_set),
                             list(model_under_learning.call_set),
                             list(model_under_learning.return_set))

    sul = VpaSUL(model_under_learning, include_top=False, check_balance=False)

    eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=100000)
    model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                   print_level=2, cex_processing='exponential_fwd')

    sul_model = SevpaSUL(model, include_top=False, check_balance=False)

    random_word_list = model.gen_random_accepting_word_bfs(min_word_length=3, amount_words=10)
    for random_word in random_word_list:
        out_model = sul_model.query(random_word)[-1]
        out_sul = sul.query(random_word)[-1]
        assert out_model == out_sul and out_model

    print(f'All tests passed for gen_random_accepting_word_bfs')

    total_len = 0
    for i in range(0, 100):
        random_word = model_under_learning.gen_random_accepting_word(return_letter_prob=0.5)
        total_len += len(random_word)
        out_model = sul_model.query(random_word)[-1]
        out_sul = sul.query(random_word)[-1]
        assert out_model == out_sul and out_model

    print(f'All tests passed average word length: {total_len / 100}')

    total_len = 0
    for i in range(0, 100):
        random_word = model_under_learning.gen_random_accepting_word(return_letter_prob=0.5, early_finish=False)
        total_len += len(random_word)
        out_model = sul_model.query(random_word)[-1]
        out_sul = sul.query(random_word)[-1]
        assert out_model == out_sul and out_model

    print(f'All tests passed average word length: {total_len / 100}')

    total_len = 0
    for i in range(0, 100):
        random_word = model_under_learning.gen_random_accepting_word(early_finish=False)
        total_len += len(random_word)
        out_model = sul_model.query(random_word)[-1]
        out_sul = sul.query(random_word)[-1]
        assert out_model == out_sul and out_model

    print(f'All tests passed average word length: {total_len / 100}')


def visual_test_to_state_setup_sevpa():
    vpa = vpa_for_L11()

    alphabet = SevpaAlphabet(list(vpa.internal_set),
                             list(vpa.call_set),
                             list(vpa.return_set))
    sul = VpaSUL(vpa, include_top=False, check_balance=False)
    eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=100000)
    model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                   print_level=2, cex_processing='rs')

    error_states = model.find_error_states()
    if error_states is not None:
        model.delete_state(error_states[0])

    # for visual comparison
    model.visualize()
    state_setup_dict = model.to_state_setup()
    model_from_setup = Sevpa.from_state_setup(state_setup_dict, "q0", alphabet)
    visualize_automaton(model_from_setup, 'ModelFromSetup')


def test_cex_processing_strategies_vpa():
    cex_processing_strategies = ['linear_fwd', 'linear_bwd', 'exponential_fwd', 'exponential_bwd', 'rs']

    for i, vpa in enumerate(
            [vpa_for_L1(), vpa_for_L2(), vpa_for_L3(), vpa_for_L4(), vpa_for_L5(), vpa_for_L7(), vpa_for_L8(),
             vpa_for_L9(), vpa_for_L10(), vpa_for_L11(), vpa_for_L12(), vpa_for_L13(), vpa_for_L14(), vpa_for_L15()]):

        print(f'VPA {i + 1 if i < 6 else i + 2}')

        model_under_learning = vpa

        alphabet = SevpaAlphabet(list(model_under_learning.internal_set),
                                 list(model_under_learning.call_set),
                                 list(model_under_learning.return_set))

        for cex_processing in cex_processing_strategies:
            sul = VpaSUL(model_under_learning, include_top=False, check_balance=False)
            eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=100000)
            model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                           print_level=1, cex_processing=cex_processing)

            error_states = model.find_error_states()
            if error_states:
                model.delete_state(error_states[0])
            sul_learned_model = SevpaSUL(model, include_top=False, check_balance=False)

            print(f'Checking {cex_processing}')
            for i in range(0, 500000):
                word_length = random.randint(1, 100)
                word = []
                for j in range(0, word_length):
                    word.append(random.choice(alphabet.get_merged_alphabet()))

                vpa_out = sul.query(tuple(word))
                learned_model_out = sul_learned_model.query(tuple(word))

                if vpa_out == learned_model_out:
                    continue
                else:
                    print(f'{cex_processing} failed on following test:')
                    print(f'Input: {word}')
                    print(f'Vpa out: {vpa_out} \nLearned vpa out: {learned_model_out}')
                    break


test_random_word_gen()
exit()

# test_arithmetic_expression()
# import cProfile
# pr = cProfile.Profile()
# pr.enable()
# test_on_random_svepa()
# pr.disable()
# pr.print_stats(sort='tottime')
# exit()


for i, vpa in enumerate(
        [vpa_for_L1(), vpa_for_L2(), vpa_for_L3(), vpa_for_L4(), vpa_for_L5(), vpa_for_L7(), vpa_for_L8(),
         vpa_for_L9(), vpa_for_L10(), vpa_for_L11(), vpa_for_L12(), vpa_for_L13(), vpa_for_L14(), vpa_for_L15()]):

    print(f'VPA {i + 1 if i < 6 else i + 2}')
    # 16 works
    for s in range(10):
        print(s)
        seed(s)
        model_under_learning = vpa

        alphabet = SevpaAlphabet(list(model_under_learning.internal_set),
                                 list(model_under_learning.call_set),
                                 list(model_under_learning.return_set))

        # if i == 9:
        #    alphabet.exclusive_call_return_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}

        sul = VpaSUL(model_under_learning, include_top=False, check_balance=False)

        eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000)
        # model = run_KV_vpda(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3,)
        model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                       print_level=2, cex_processing='linear_bwd')

        e = model.gen_random_accepting_word_bfs(min_word_length=5, amount_words=100)
        print(e[:25])
        print('generate random acc')
        for _ in range(10):
            print(model.gen_random_accepting_word())
