import ast
import random

from Examples import learning_context_free_grammar_example
from aalpy.SULs.AutomataSUL import SevpaSUL
from aalpy.base import SUL
from aalpy.learning_algs import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle, StatePrefixEqOracle
from aalpy.utils import visualize_automaton, get_Angluin_dfa, generate_random_sevpa
from aalpy.utils.BenchmarkSevpaModels import *
from random import seed


# learning_context_free_grammar_example()

# TODOs
# 1. exponential cex processing in CounterExampleProcessing.py
# 2. Create a SEVPA function that generates random positive strings - model.generate_random_positive_string()
# 2. Add all 15 langs as SVEPA
# 4. Implement and test to_state_setup, test saving and loading to/from file
# 5. Create an active interface to learn a grammar of some language, like simplified C or Java

# Thesis
# 1. Intro
# 2. Preliminaries (very important)
# 2.1 CFG, context pairs, well matched words
# 2.2 What are SEVPA and why we use those instead of VPAs
# 2.3 Example SEVPA and how to read/interpret it (Important on a small example)
# 2.4 Automata Learning and KV
# ...
# 3. KV for CFG inference (intuition behind everything and how it fits with preliminaries)
# 3.1 Explain alg in detail, like Maxi
# 3.2 Explain CEX processing/transform access string, also on example and intuition
# 3.3 Important: Run of the algorithm, visualize classification tree...
# 4. Evaluation
# - number of steps/queries for models of growing alphabet, state size, ...]
# - on 15 languages
# - on random languages
# - on something cool

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


def test_on_random_svepa():
    random_svepa = generate_random_sevpa(num_states=50, internal_alphabet_size=3,
                                         call_alphabet_size=3,
                                         return_alphabet_size=3,
                                         acceptance_prob=0.4,
                                         return_transition_prob=0.5)

    alphabet = random_svepa.input_alphabet

    sul = SevpaSUL(random_svepa)

    eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000,
                                   min_walk_len=10, max_walk_len=30)
    # model = run_KV_vpda(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3,)
    model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                   print_level=2, cex_processing='rs')


def test_cex_processing_strategies_vpa():
    cex_processing_strategies = ['rs', 'linear_fwd', 'linear_bwd', 'exponential_fwd', 'exponential_bwd', ]

    for i, vpa in enumerate(
            [sevpa_for_L1(), sevpa_for_L2(), sevpa_for_L3(), sevpa_for_L4(), sevpa_for_L5(), sevpa_for_L7(), sevpa_for_L8(),
             sevpa_for_L9(), sevpa_for_L10(), sevpa_for_L11(), sevpa_for_L12(), sevpa_for_L13(), sevpa_for_L14(), sevpa_for_L15()]):

        print(f'VPA {i + 1 if i < 6 else i + 2}')

        model_under_learning = vpa

        alphabet = SevpaAlphabet(list(model_under_learning.internal_set),
                                 list(model_under_learning.call_set),
                                 list(model_under_learning.return_set))

        for cex_processing in cex_processing_strategies:
            sul = SevpaSUL(model_under_learning)
            eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=20000)
            model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                           print_level=1, cex_processing=cex_processing)

            sul_learned_model = SevpaSUL(model)

            print(f'Checking {cex_processing}')
            for i in range(0, 10000):
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
                    assert False

# test_cex_processing_strategies_vpa()
# test_arithmetic_expression()
# test_on_random_svepa()
# import cProfile
# pr = cProfile.Profile()
# pr.enable()
# test_on_random_svepa()
# pr.disable()
# pr.print_stats(sort='tottime')
# exit()


import re

expression = "baadsf / ('q1', '(fasfas')"

# Define the regex pattern
pattern = r"(\S+)\s*/\s*\(\s*'(\S+)'\s*,\s*'(\S+)'\s*\)"

# Match the pattern in the expression
match = re.match(r"(\S+)\s*/\s*\(\s*'(\S+)'\s*,\s*'(\S+)'\s*\)", expression)

# Extract groups if there is a match
if match:
    a, b, c = match.groups()
    print("a:", a)
    print("b:", b)
    print("c:", c)
else:
    print("No match found.")

exit()

for i, vpa in enumerate(
        [sevpa_for_L1(), sevpa_for_L2(), sevpa_for_L3(), sevpa_for_L4(), sevpa_for_L5(), sevpa_for_L7(), sevpa_for_L8(),
         sevpa_for_L9(), sevpa_for_L10(), sevpa_for_L11(), sevpa_for_L12(), sevpa_for_L13(), sevpa_for_L14(),
         sevpa_for_L15()]):

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

        sul = SevpaSUL(model_under_learning)

        eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000)
        # model = run_KV_vpda(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3,)
        model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                       print_level=2, cex_processing='exponential_fwd')
