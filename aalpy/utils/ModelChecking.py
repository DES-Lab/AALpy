import os
import re
import aalpy.paths
from collections import defaultdict

from aalpy.automata import Mdp, StochasticMealyMachine

prism_prob_output_regex = re.compile("Result: (\d+\.\d+)")


def get_properties_file(exp_name):
    property_files = {
        'first_grid': aalpy.paths.path_to_properties + 'first_eval.props',
        'second_grid': aalpy.paths.path_to_properties + 'second_eval.props',
        'shared_coin': aalpy.paths.path_to_properties + 'shared_coin_eval.props',
        'slot_machine': aalpy.paths.path_to_properties + 'slot_machine_eval.props',
        'mqtt': aalpy.paths.path_to_properties + 'emqtt_two_client.props',
        'tcp': aalpy.paths.path_to_properties + 'tcp_eval.props'
    }
    return property_files[exp_name]


def get_correct_prop_values(exp_name):
    correct_model_properties = {
        'first_grid': {'prob1': 0.96217534, 'prob2': 0.6499274956800001, 'prob3': 0.6911765746880001},
        'second_grid': {'prob1': 0.93480795088125, 'prob2': 0.6711947700000002, 'prob3': 0.9742903305241055,
                        'prob4': 0.14244219329051103},
        'shared_coin': {'prob1': 0.10694382182657244, 'prob2': 0.5555528623795738, 'prob3': 0.3333324384052837,
                        'prob4': 0.42857002816478273, 'prob5': 0.001708984375, 'prob6': 0.266845703125,
                        'prob7': 0.244384765625, 'prob8': 0.263427734375},
        'slot_machine': {'prob1': 0.36380049887344645, 'prob2': 0.6445910164135946, 'prob3': 1.0, 'prob4': 0.159,
                         'prob5': 0.28567, 'prob6': 0.2500000000000001, 'prob7': 0.025445087448668406},
        'mqtt': {'prob1': 0.9612, 'prob2': 0.34390000000000004, 'prob3': 0.6513215599000001, 'prob4': 0.814697981114816,
                 'prob5': 0.7290000000000001},
        'tcp': {'prob1': 0.19, 'prob2': 0.5695327900000001, 'prob3': 0.7712320754503901, 'prob4': 0.8784233454094308}
    }
    return list(correct_model_properties[exp_name].values())


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


def evaluate_all_properties(prism_file_name, properties_file_name):
    import subprocess
    import io
    from os import path

    prism_file = aalpy.paths.path_to_prism.split('/')[-1]
    path_to_prism_file = aalpy.paths.path_to_prism[:-len(prism_file)]

    file_abs_path = path.abspath(prism_file_name)
    properties_als_path = path.abspath(properties_file_name)
    results = {}
    proc = subprocess.Popen(
        [aalpy.paths.path_to_prism, file_abs_path, properties_als_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=path_to_prism_file)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        if not line:
            break
        else:
            match = prism_prob_output_regex.match(line)
            if match:
                results[f'prop{len(results) + 1}'] = float(match.group(1))
    proc.kill()
    return results


def model_check_properties(model: Mdp, properties: str):
    """

    Args:
        model: Markov Decision Process that serves as a basis for model checking.
        properties: Properties file. It should point to a file under the path_to_properties folder.

    Returns:

        results of model checking
    """
    from os import remove
    from aalpy.utils import mdp_2_prism_format
    mdp_2_prism_format(mdp=model, name='mc_exp', output_path=f'mc_exp.prism')

    prism_model_path = f'mc_exp.prism'

    data = evaluate_all_properties(prism_model_path, properties)

    remove(prism_model_path)

    return data


def model_check_experiment(path_to_properties, correct_prop_values, mdp, precision=4):
    """
    For our stochastic experiments you can use this function.
    For example, check learn_stochastic_system_and_do_model_checking in Examples.py

    Args:
        path_to_properties: path to the properties file
        correct_prop_values: correct values of all properties. In list, where property at index i corresponds to the
            i-th element of the list.
        mdp: MDP
        precision: precision to which round up results

    Returns:

        results of model checking and absolute differance to the correct results
    """
    model_checking_results = model_check_properties(mdp, path_to_properties)

    diff_2_correct = dict()
    for ind, val in enumerate(model_checking_results.values()):
        diff_2_correct[f'prop{ind+1}'] = round(abs(correct_prop_values[ind] - val), precision)

    results = {key: round(val, precision) for key, val in model_checking_results.items()}
    return results, diff_2_correct


def stop_based_on_confidence(hypothesis, property_based_stopping, print_level=2):
    """

    Args:

        hypothesis: Markov decision process
        property_based_stopping: a tuple (path to properties file, list of correct property values, max allowed error)
        print_level: 2 or 3 if output of model checking is to be printed during learning

    Returns:

        True if absolute error for all properties is smaller then property_based_stopping[2]
    """
    from aalpy.utils import smm_to_mdp_conversion

    path_2_prop = property_based_stopping[0]
    correct_values = property_based_stopping[1]
    error_bound = property_based_stopping[2]

    model = hypothesis
    if isinstance(hypothesis, StochasticMealyMachine):
        model = smm_to_mdp_conversion(hypothesis)

    res, diff = model_check_experiment(path_2_prop, correct_values, model)

    if print_level >= 2:
        print('Error for each property:', [round(d * 100, 2) for d in diff.values()])
    if not diff:
        return False

    for d in diff.values():
        if d > error_bound:
            return False

    return True
