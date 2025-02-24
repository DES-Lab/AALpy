import pickle
from collections import defaultdict
from random import shuffle

from aalpy import run_RPNI, run_PAPNI, AutomatonSUL
from aalpy.utils import convert_i_o_traces_for_RPNI, generate_input_output_data_from_vpa
from aalpy.utils.BenchmarkVpaModels import get_all_VPAs
from statistics import mean, stdev

all_data = dict()
with open('papni_sequances.pickle', 'rb') as handle:
    all_data = pickle.load(handle)


def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = calculate_f1_score(precision, recall)

    return precision, recall, f1


def compare_rpni_and_papni(test_data, rpni_model, papni_model):
    def evaluate_model(learned_model, test_data):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for input_seq, correct_output in test_data:
            learned_model.reset_to_initial()
            learned_output = learned_model.execute_sequence(learned_model.initial_state, input_seq)[-1]

            if learned_output and correct_output:
                true_positives += 1
            elif learned_output and not correct_output:
                false_positives += 1
            elif not learned_output and correct_output:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = calculate_f1_score(precision, recall)

        return precision, recall, f1

    rpni_error = evaluate_model(rpni_model, test_data)
    papni_error = evaluate_model(papni_model, test_data)

    # print(f'RPNI size {rpni_model.size} vs {papni_model.size} PAPNI size')
    # print(f'RPNI   precision, recall, f1: {rpni_error}')
    # print(f'PAPNI  precision, recall, f1: {papni_error}')

    return [rpni_model.size, papni_model.size, rpni_error, papni_error]


def get_sequances_from_active_sevpa(model):
    from aalpy import SUL, run_KV, RandomWordEqOracle, SevpaAlphabet

    class CustomSUL(SUL):
        def __init__(self, automatonSUL):
            super(CustomSUL, self).__init__()
            self.sul = automatonSUL
            self.sequances = []

        def pre(self):
            self.tc = []
            self.sul.pre()

        def post(self):
            self.sequances.append(self.tc)
            self.sul.post()

        def step(self, letter):
            output = self.sul.step(letter)
            if letter is not None:
                self.tc.append((letter, output))
            return output

    vpa_alphabet = model.get_input_alphabet()
    alphabet = SevpaAlphabet(vpa_alphabet.internal_alphabet, vpa_alphabet.call_alphabet, vpa_alphabet.return_alphabet)
    sul = AutomatonSUL(model)
    sul = CustomSUL(sul)
    eq_oracle = RandomWordEqOracle(alphabet.get_merged_alphabet(), sul, num_walks=50000, min_walk_len=6,
                                   max_walk_len=30, reset_after_cex=True)
    # eq_oracle = BreadthFirstExplorationEqOracle(vpa_alphabet.get_merged_alphabet(), sul, 7)
    _ = run_KV(alphabet, sul, eq_oracle, automaton_type='vpa', print_level=3)

    return convert_i_o_traces_for_RPNI(sul.sequances)


def split_data_to_learning_and_testing(data, learning_to_test_ratio=0.5):
    total_number_positive = len([x for x in data if x[1]])
    total_number_negative = len(data) - total_number_positive

    num_learning_positive_seq = total_number_positive * learning_to_test_ratio
    num_learning_negative_seq = total_number_negative * learning_to_test_ratio

    # sorted(data, key=lambda x: len(x[0]))
    shuffle(data)

    learning_sequances, test_sequances = [], []

    l_pos, l_neg = 0, 0
    for seq, label in data:
        if label and l_pos <= num_learning_positive_seq:
            learning_sequances.append((seq, label))
            l_pos += 1
        elif not label and l_neg <= num_learning_negative_seq:
            learning_sequances.append((seq, label))
            l_neg += 1
        else:
            test_sequances.append((seq, label))

    return learning_sequances, test_sequances


def run_experiment(experiment_id,
                   ground_truth_model,
                   num_of_learning_seq,
                   max_learning_seq_len,
                   random_data_generation=True,
                   learning_to_test_ratio=0.5):
    if random_data_generation:
        data = generate_input_output_data_from_vpa(ground_truth_model,
                                                   num_sequances=num_of_learning_seq,
                                                   max_seq_len=max_learning_seq_len,
                                                )
    else:
        all_generated_data = all_data[experiment_id]

        # sorted(all_generated_data, key=lambda x: len(x))
        shuffle(all_generated_data)

        positive_seq = [x for x in all_generated_data if x[1]]
        negative_seq = [x for x in all_generated_data if not x[1]]

        data = []
        data += positive_seq[:5000]
        data += negative_seq[:10000 - len(data)]

        # wm_negative = 0
        # for seq, label in data:
        #     if not label and ground_truth_model.is_balanced(seq):
        #         wm_negative += 1
        # print(wm_negative)

        # data = get_sequances_from_active_sevpa(ground_truth_model)

    vpa_alphabet = ground_truth_model.get_input_alphabet()

    learning_data, test_data = split_data_to_learning_and_testing(data, learning_to_test_ratio=learning_to_test_ratio)

    num_positive_learning = len([x for x in learning_data if x[1]])
    learning_set_size = (num_positive_learning, len(learning_data) - num_positive_learning)

    num_positive_test = len([x for x in test_data if x[1]])
    num_test_size = (num_positive_test, len(test_data) - num_positive_test)

    rpni_model = run_RPNI(learning_data, 'dfa', print_info=False, input_completeness='sink_state')

    papni_model = run_PAPNI(learning_data, vpa_alphabet, print_info=False)

    comparison_results = compare_rpni_and_papni(test_data, rpni_model, papni_model)

    comparison_results = comparison_results + [learning_set_size, num_test_size]
    return comparison_results


def run_all_experiments_experiments(test_models, learning_to_test_ratio):
    for idx, gt in enumerate(test_models):
        results = run_experiment(idx, gt, num_of_learning_seq=10000, max_learning_seq_len=50,
                                 random_data_generation=False, learning_to_test_ratio=learning_to_test_ratio)

        res_str = f'GT {idx + 1}:\t Learning ({results[-2][0]}/{results[-2][1]}),\t Test ({results[-1][0]}/{results[-1][1]}),\t'
        res_str += f'RPNI: size: {results[0]}, prec/rec/F1: {results[2]}, \t PAPNI size: {results[1]}, prec/rec/F1: {results[3]}'

        print(res_str)


def run_experiments_multiple_times(test_models, num_times):
    all_results = defaultdict(list)
    for idx, gt in enumerate(test_models):
        for _ in range(num_times):
            r = run_experiment(idx, gt, num_of_learning_seq=10000, max_learning_seq_len=50,
                               random_data_generation=False, learning_to_test_ratio=0.5)

            all_results[idx].append(r)

    for exp_idx, result in all_results.items():
        print('Experiment: ', exp_idx)
        rpni_results = [r[2] for r in result]
        papni_results = [r[3] for r in result]

        rpni_f1 = [r[2] for r in rpni_results]
        papni_f1 = [r[2] for r in papni_results]

        print(f'RPNI  : min {min(rpni_f1)}, max {max(rpni_f1)}, mean {mean(rpni_f1)}, stddev {stdev(rpni_f1)}')
        print(f'PAPNI : min {min(papni_f1)}, max {max(papni_f1)}, mean {mean(papni_f1)}, stddev {stdev(papni_f1)}')
        print('----------------------------------------------------------------')

    import pickle
    with open('papni_rpni_eval_results.pickle', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


all_models = get_all_VPAs()

run_all_experiments_experiments(all_models, learning_to_test_ratio=0.5)
# run_experiments_multiple_times(all_models, 20)
