from aalpy import run_RPNI, run_PAPNI, load_automaton_from_file
from aalpy.utils import convert_i_o_traces_for_RPNI, generate_input_output_data_from_vpa
from aalpy.utils.BenchmarkVpaModels import get_all_VPAs


def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = calculate_f1_score(precision, recall)

    return precision, recall, f1


def compare_rpni_and_papni(original_model, rpni_model, papni_model, num_sequances, min_seq_len, max_seq_len):
    test_data = generate_input_output_data_from_vpa(original_model, num_sequances, min_seq_len, max_seq_len)
    test_data = convert_i_o_traces_for_RPNI(test_data)

    def calculate_f1_score(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_model(learned_model, test_data):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for input_seq, out in test_data:
            learned_model.reset_to_initial()
            learned_output = learned_model.execute_sequence(learned_model.initial_state, input_seq)[-1]

            if learned_output and out:
                true_positives += 1
            elif learned_output and not out:
                false_positives += 1
            elif not learned_output and out:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = calculate_f1_score(precision, recall)

        return precision, recall, f1

    rpni_error = evaluate_model(rpni_model, test_data)
    papni_error = evaluate_model(papni_model, test_data)

    print(f'-----------------------------------------------------------------')
    print(f'RPNI size {rpni_model.size} vs {papni_model.size} PAPNI size')
    print(f'RPNI   precision, recall, f1: {rpni_error}')
    print(f'PAPNI  precision, recall, f1: {papni_error}')


arithmetics_model = load_automaton_from_file('../DotModels/arithmetics.dot', 'vpa')

# 15 test benchmarks
test_models = [arithmetics_model]
test_models.extend(get_all_VPAs())

for ground_truth in test_models:
    vpa_alphabet = ground_truth.get_input_alphabet()

    data = generate_input_output_data_from_vpa(ground_truth, num_sequances=2000, min_seq_len=1, max_seq_len=12)
    data = convert_i_o_traces_for_RPNI(data)

    rpni_model = run_RPNI(data, 'dfa', print_info=True, input_completeness='sink_state')

    papni_model = run_PAPNI(data, vpa_alphabet, print_info=True)

    compare_rpni_and_papni(ground_truth, rpni_model, papni_model, num_sequances=100, min_seq_len=20, max_seq_len=40)
