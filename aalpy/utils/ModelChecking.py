import os
import re
from collections import defaultdict

from aalpy.automata import Mdp

prism_prob_output_regex = re.compile("Result: (.+?) \\(.*\\)")


def _target_string(target, orig_id_to_int_id):
    target_state = target[0]
    target_prob = target[1]
    target_id = orig_id_to_int_id[target_state.state_id]
    return f"{target_prob} : (loc'={target_id})"


def _sanitize_for_prism(symbol):
    if symbol in ["mdp", "init", "module", "endmodule", "label"]:
        return "___" + symbol + "___"
    else:
        return symbol


def mdp_2_prism_format(mdp: Mdp, name: str, output_path=None):
    """
    Translates MDP to Prims modelling language.

    Args:

        mdp: markov decision process

        name: name of the mdp/experiment

        output_path: output file (Default value = None)

    """
    module_string = "mdp"
    module_string += os.linesep
    module_string += f"module {name}"
    module_string += os.linesep

    nr_states = len(mdp.states)
    orig_id_to_int_id = dict()
    for i, s in enumerate(mdp.states):
        orig_id_to_int_id[s.state_id] = i
    module_string += "loc : [0..{}] init {};".format(nr_states, orig_id_to_int_id[mdp.initial_state.state_id])
    module_string += os.linesep

    # print transitions
    for source in mdp.states:
        source_id = orig_id_to_int_id[source.state_id]
        for inp in source.transitions.keys():
            if source.transitions[inp]:
                target_strings = \
                    map(lambda target: _target_string(target, orig_id_to_int_id), source.transitions[inp])
                target_joined = " + ".join(target_strings)
                module_string += f"[{_sanitize_for_prism(inp)}] loc={source_id} -> {os.linesep} {target_joined};"
                module_string += os.linesep
    module_string += "endmodule"
    module_string += os.linesep
    # labelling function
    output_to_state_id = defaultdict(list)
    for s in mdp.states:
        joined_output = s.output
        outputs = joined_output.split("__")
        for o in outputs:
            if o:
                output_to_state_id[o].append(orig_id_to_int_id[s.state_id])

    for output, states in output_to_state_id.items():
        state_propositions = map(lambda s_id: "loc={}".format(s_id), states)
        state_disjunction = "|".join(state_propositions)
        output_string = _sanitize_for_prism(output)
        module_string += f"label \"{output_string}\" = {state_disjunction};"
        module_string += os.linesep

    if output_path:
        with open(output_path, "w") as text_file:
            text_file.write(module_string)
    return module_string


def count_properties(properties_content):
    return len(list(filter(lambda line: len(line.strip()) > 0, properties_content)))


def properties_string(properties_sorted, property_data, property_data_original):
    property_string = ""
    for p in properties_sorted:
        distance = abs(property_data[p] - property_data_original[p])
        property_string += "," + "{:.4f}".format(property_data[p]) + "|{:.4f}".format(distance)
    return property_string


def eval_property(prism_executable, prism_file_name, properties_file_name, property_index):
    import subprocess
    import io
    from os import path

    prism_file = prism_executable.split('/')[-1]
    path_to_prism_file = prism_executable[:-len(prism_file)]

    file_abs_path = path.abspath(prism_file_name)
    properties_als_path = path.abspath(properties_file_name)

    proc = subprocess.Popen(
        [prism_file, file_abs_path, properties_als_path, "-prop", str(property_index)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=path_to_prism_file)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        if not line:
            break
        else:
            match = prism_prob_output_regex.match(line)
            if match:
                return float(match.group(1))
    return 0.0


def eval_properties(prism_executable, prism_file_name, properties_file_name):
    data = dict()
    with open(properties_file_name, 'r') as properties_file:
        properties_content = properties_file.readlines()
        nr_properties = count_properties(properties_content)
        for property_index in range(1, nr_properties + 1):
            probability = eval_property(prism_executable, prism_file_name, properties_file_name,
                                        property_index)
            data[f"prob{property_index}"] = probability

    return data


def model_check_with_prism(path_to_prism: str, model: Mdp, exp_name, properties: str):
    from os import remove
    from aalpy.utils import mdp_2_prism_format
    mdp_2_prism_format(mdp=model, name=exp_name, output_path=f'{exp_name}.prism')

    prism_model_path = f'{exp_name}.prism'

    data = eval_properties(path_to_prism, prism_model_path, properties)

    remove(prism_model_path)

    return data


def model_check_experiment(path_to_prism, exp_name, mdp):
    assert exp_name in ['first_grid', 'second_grid', 'shared_coin', 'slot_machine', 'mqtt', 'tcp']

    folder = 'Benchmarking/prism_eval_props/'
    property_files = {
        'first_grid': folder + 'first_eval.props',
        'second_grid': folder + 'second_eval.props',
        'shared_coin': folder + 'shared_coin_eval.props',
        'slot_machine': folder + 'slot_machine_eval.props',
        'mqtt': folder + 'emqtt_two_client.props',
        'tcp': folder + 'tcp_eval.props'
    }

    correct_model_properties = {
        'first_grid': {'prob1': 0.96217534, 'prob2': 0.6499274956800001, 'prob3': 0.6911765746880001},
        'second_grid': {'prob1': 0.93480795088125, 'prob2': 0.6711947700000002, 'prob3': 0.9742903305241055,
                        'prob4': 0.14244219329051103},
        'shared_coin': {'prob1': 0.10694382182657244, 'prob2': 0.5555528623795738, 'prob3': 0.3333324384052837,
                        'prob4': 0.42857002816478273, 'prob5': 0.001708984375, 'prob6': 0.266845703125,
                        'prob7': 0.244384765625, 'prob8': 0.263427734375},
        'slot_machine': {'prob1': 0.36380049887344645, 'prob2': 0.6445910164135946, 'prob3': 1.0, 'prob4': 0.0,
                         'prob5': 0.0, 'prob6': 0.2500000000000001, 'prob7': 0.0},
        'mqtt': {'prob1': 0.9612, 'prob2': 0.34390000000000004, 'prob3': 0.6513215599000001, 'prob4': 0.814697981114816,
                 'prob5': 0.7290000000000001},
        'tcp': {'prob1': 0.19, 'prob2': 0.5695327900000001, 'prob3': 0.7712320754503901, 'prob4': 0.8784233454094308}
    }

    model_checking_results = model_check_with_prism(path_to_prism, mdp, exp_name, property_files[exp_name])

    diff_2_correct = dict()
    for prop, val in model_checking_results.items():
        diff_2_correct[prop] = abs(correct_model_properties[exp_name][prop] - val)

    return diff_2_correct
