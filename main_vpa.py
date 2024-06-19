import ast
import random

from aalpy import SUL, SevpaAlphabet
from aalpy.learning_algs.vpa_passive.VpaPTA import check_vpa_sequence, extract_unique_sequences, \
    create_Vpa_PTA
from aalpy.learning_algs.vpa_passive.vpa_RPNI import RPNI_VPA
from aalpy.utils import convert_i_o_traces_for_RPNI
from aalpy.utils.BenchmarkVpaModels import *
from aalpy.utils import generate_random_sevpa

def get_list_data():

    alphabet = VpaAlphabet(internal_alphabet=['1'], call_alphabet=['(', '['], return_alphabet=[')', ']'])

    rpni_format = [
        # (tuple(), False),
        (('[', '[', ']', ']'), True),
        (('[', '[', ']', ')'), False),
        (('[', '[', ')', ']'), False),
        (('[', ')'), False),
        (('[', '[', ']', '1', ']'), True),
        (('[', '1', '[', ']', '1', ']'), True),
        (('(', '1', '[', ']', '1', ')'), True),
        (('(', ')'), True),
        (('[', ']'), True),
        (('[', '1', '(', ']', '1', ')'), False),
    ]

    return rpni_format, alphabet


def gen_aritmetic_data(num_sequances=3000, min_seq_len=2, max_seq_len=8):
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
                self.string_under_test += ' ' + letter if len(self.string_under_test) > 0 else letter

            try:
                # Parse the expression using ast
                parsed_expr = ast.parse(self.string_under_test, mode='eval')
                # Check if the parsed expression is a valid arithmetic expression
                is_valid = all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Name, ast.Load))
                               or isinstance(node, ast.operator) or isinstance(node, ast.expr_context)
                               or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.operator))
                               for node in ast.walk(parsed_expr))
                return is_valid

            except SyntaxError:
                return False

        def is_balanced(self, seq, alphabet):
            counter = 0
            for i in seq:
                if i in alphabet.call_alphabet:
                    counter += 1
                if i in alphabet.return_alphabet:
                    counter -= 1
                if counter < 0:
                    return False
            return counter == 0

    sul = ArithmeticSUL()

    alphabet = VpaAlphabet(internal_alphabet=['1', '+', '-', '/', '0'], call_alphabet=['(', '['], return_alphabet=[')'])
    merged_alph = alphabet.get_merged_alphabet()
    data = []
    while len(data) < num_sequances:
        seq = []
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            seq.append(random.choice(merged_alph))

            outputs = sul.query(seq)

            if outputs[-1] is False:
                continue

            data.append(list(zip(seq, outputs)))

    rpni_format = convert_i_o_traces_for_RPNI(data)

    # TODO fix later also for deterministic regular
    rpni_format.append(((), False))

    counter = 0
    for seq, x in rpni_format.copy():
        if x:
            counter += 1
        if not sul.is_balanced(seq, alphabet):
            rpni_format.remove((seq, x))

    print(f'# Positive: {counter}')
    print(f'# Negative: {len(rpni_format) - counter}')
    return rpni_format, alphabet

def gen_aritmetic_data(num_sequances=3000, min_seq_len=2, max_seq_len=8):
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
                self.string_under_test += ' ' + letter if len(self.string_under_test) > 0 else letter

            try:
                # Parse the expression using ast
                parsed_expr = ast.parse(self.string_under_test, mode='eval')
                # Check if the parsed expression is a valid arithmetic expression
                is_valid = all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Name, ast.Load))
                               or isinstance(node, ast.operator) or isinstance(node, ast.expr_context)
                               or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.operator))
                               for node in ast.walk(parsed_expr))
                return is_valid

            except SyntaxError:
                return False

        def is_balanced(self, seq, alphabet):
            counter = 0
            for i in seq:
                if i in alphabet.call_alphabet:
                    counter += 1
                if i in alphabet.return_alphabet:
                    counter -= 1
                if counter < 0:
                    return False
            return counter == 0

    sul = ArithmeticSUL()

    alphabet = VpaAlphabet(internal_alphabet=['1', '+', '-', '/', '0'], call_alphabet=['(', '['], return_alphabet=[')'])
    merged_alph = alphabet.get_merged_alphabet()
    data = []
    while len(data) < num_sequances:
        seq = []
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            seq.append(random.choice(merged_alph))

            outputs = sul.query(seq)

            if outputs[-1] is False:
                continue

            data.append(list(zip(seq, outputs)))

    rpni_format = convert_i_o_traces_for_RPNI(data)

    # TODO fix later also for deterministic regular
    rpni_format.append(((), False))

    counter = 0
    for seq, x in rpni_format.copy():
        if x:
            counter += 1
        if not sul.is_balanced(seq, alphabet):
            rpni_format.remove((seq, x))

    print(f'# Positive: {counter}')
    print(f'# Negative: {len(rpni_format) - counter}')
    return rpni_format, alphabet


def gen_data(vpa, num_sequances=2000, min_seq_len=2, max_seq_len=4):
    alphabet = vpa.input_alphabet.get_merged_alphabet()
    data = []
    while len(data) < num_sequances:
        seq = []
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            seq.append(random.choice(alphabet))

        if not vpa.is_balanced(seq):
            continue

        vpa.reset()
        outputs = vpa.execute_sequence(vpa.initial_state, seq)

        if outputs[-1] == False:
            continue
        data.append(list(zip(seq, outputs)))

    rpni_format = convert_i_o_traces_for_RPNI(data)

    # TODO fix later also for deterministic regular
    rpni_format.append(((), vpa.initial_state.is_accepting))

    counter = 0
    for seq, x in rpni_format.copy():
        if x:
            counter += 1
        if not vpa.is_balanced(seq):
            rpni_format.remove((seq, x))

    print(f'# Positive: {counter}')
    print(f'# Negative: {len(rpni_format) - counter}')
    return rpni_format


# random.seed(1)
# #
# gt = vpa_for_L7()
# vpa_alphabet = gt.input_alphabet

# data = gen_data(gt, num_sequances=1500, min_seq_len=2, max_seq_len=8)

# data, vpa_alphabet = gen_aritmetic_data(num_sequances=2000, min_seq_len=2, max_seq_len=15)

data, vpa_alphabet = get_list_data()

pta = create_Vpa_PTA(data, vpa_alphabet)

test_seq = extract_unique_sequences(pta)
for d in test_seq:
    check_vpa_sequence(pta, d, vpa_alphabet)

rpni = RPNI_VPA(data, vpa_alphabet, print_info=True)
learned_model = rpni.run_vpa_rpni()
learned_model.visualize()

for seq, o in data:
    if not seq:
        continue
    learned_model.reset()
    learned_output = learned_model.execute_sequence(learned_model.initial_state, seq)[-1]
    if o != learned_output:
        print(seq, o, learned_output)
        learned_output = learned_model.execute_sequence(learned_model.initial_state, seq)[-1]

        assert False, 'Learned Model not consistent with data'

print('ALL GOOD')
