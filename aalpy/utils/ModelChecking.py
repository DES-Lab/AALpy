# Model checking utilities: PRISM export, property evaluation, bisimilarity and test-case generation.
import itertools as it
import os
import re
from collections import defaultdict
from queue import Queue
from random import choices
from typing import Any

import aalpy.paths
from aalpy.SULs import AutomatonSUL
from aalpy.automata import Mdp, StochasticMealyMachine, MealyMachine, Dfa, MooreMachine, MooreState, MealyState, \
    DfaState
from aalpy.base import DeterministicAutomaton, SUL, AutomatonState

prism_prob_output_regex = re.compile("Result: (\d+\.\d+)")


def get_properties_file(exp_name: str) -> str:
    """
    Looks up the path to the properties file of a predefined stochastic experiment.

    :param str exp_name: Name of the experiment.
    :return str: Path to the corresponding properties file.
    """
    property_files = {
        'first_grid': aalpy.paths.path_to_properties + 'first_eval.props',
        'second_grid': aalpy.paths.path_to_properties + 'second_eval.props',
        'shared_coin': aalpy.paths.path_to_properties + 'shared_coin_eval.props',
        'slot_machine': aalpy.paths.path_to_properties + 'slot_machine_eval.props',
        'mqtt': aalpy.paths.path_to_properties + 'emqtt_two_client.props',
        'tcp': aalpy.paths.path_to_properties + 'tcp_eval.props',
        'bluetooth': aalpy.paths.path_to_properties + 'bluetooth.props',
    }
    return property_files[exp_name]


def get_correct_prop_values(exp_name: str) -> list:
    """
    Looks up the reference (ground truth) property values of a predefined stochastic experiment.

    :param str exp_name: Name of the experiment.
    :return list: List of correct property values.
    """
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
        'tcp': {'prob1': 0.19, 'prob2': 0.5695327900000001, 'prob3': 0.7712320754503901, 'prob4': 0.8784233454094308},
        'bluetooth': {'prop1': 0.16800000000000004, 'prop2': 0.3926480000000001, 'prop3': 0.5572338000000001,
                      'prop4': 0.6772233874640001, 'prop5': 0.7646958490393682, 'prop6': 0.8284632739463244,
                      'prop7': 0.36000000000000004, 'prop8': 0.5904, 'prop9': 0.7902848,
                      'prop10': 0.8926258176000001, 'prop11': 0.9450244186112, 'prop12': 0.9718525023289344,
                      'prop13': 0.9855884811924145}
    }
    return list(correct_model_properties[exp_name].values())


def _target_string(target: tuple, orig_id_to_int_id: dict) -> str:
    """
    Formats a single MDP transition target as a PRISM update expression.

    :param tuple target: Tuple of (target_state, probability).
    :param dict orig_id_to_int_id: Map from original state ids to PRISM integer location ids.
    :return str: The PRISM update expression string.
    """
    target_state = target[0]
    target_prob = target[1]
    target_id = orig_id_to_int_id[target_state.state_id]
    return f"{target_prob} : (loc'={target_id})"


def _sanitize_for_prism(symbol: str) -> str:
    """
    Sanitizes a symbol that clashes with a PRISM keyword.

    :param str symbol: Symbol to sanitize.
    :return str: The original symbol, or a mangled version if it is a PRISM keyword.
    """
    if symbol in ["mdp", "init", "module", "endmodule", "label"]:
        return "___" + symbol + "___"
    else:
        return symbol


def mdp_2_prism_format(mdp: Mdp, name: str, output_path: str | None = None) -> str:
    """
    Translates MDP to Prism modelling language.

    :param Mdp mdp: Markov decision process.
    :param str name: Name of the mdp/experiment.
    :param str | None output_path: Output file to which the model is written (Default value = None).
    :return str: The Prism model as a string.
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


def evaluate_all_properties(prism_file_name: str, properties_file_name: str) -> dict[str, float]:
    """
    Runs PRISM on a model file against a properties file and collects the resulting probabilities.

    :param str prism_file_name: Path to the PRISM model file.
    :param str properties_file_name: Path to the properties file.
    :return dict[str, float]: Map from property name (e.g. 'prop1') to its computed probability.
    """
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


def model_check_properties(model: Mdp, properties: str) -> dict[str, float]:
    """
    Model checks a set of properties against an MDP using PRISM.

    :param Mdp model: Markov Decision Process that serves as a basis for model checking.
    :param str properties: Properties file. It should point to a file under the path_to_properties folder.
    :return dict[str, float]: Results of model checking.
    """
    from os import remove
    from aalpy.utils import mdp_2_prism_format
    mdp_2_prism_format(mdp=model, name='mc_exp', output_path=f'mc_exp.prism')

    prism_model_path = f'mc_exp.prism'

    data = evaluate_all_properties(prism_model_path, properties)

    remove(prism_model_path)

    return data


def model_check_experiment(path_to_properties: str, correct_prop_values: list, mdp: Mdp, precision: int = 4) -> tuple[dict, dict]:
    """
    For our stochastic experiments you can use this function.
    For example, check learn_stochastic_system_and_do_model_checking in Examples.py

    :param str path_to_properties: Path to the properties file.
    :param list correct_prop_values: Correct values of all properties. In list, where property at index i corresponds
        to the i-th element of the list.
    :param Mdp mdp: MDP.
    :param int precision: Precision to which round up results.
    :return tuple[dict, dict]: Results of model checking and absolute difference to the correct results.
    """
    model_checking_results = model_check_properties(mdp, path_to_properties)

    diff_2_correct = dict()
    for ind, val in enumerate(model_checking_results.values()):
        diff_2_correct[f'prop{ind + 1}'] = round(abs(correct_prop_values[ind] - val), precision)

    results = {key: round(val, precision) for key, val in model_checking_results.items()}
    return results, diff_2_correct


def stop_based_on_confidence(hypothesis: Any, property_based_stopping: tuple, print_level: int = 2) -> bool:
    """
    Determines whether learning should stop based on model checking confidence.

    :param Any hypothesis: Markov decision process (or StochasticMealyMachine, converted internally to an MDP).
    :param tuple property_based_stopping: A tuple (path to properties file, list of correct property values,
        max allowed error).
    :param int print_level: 2 or 3 if output of model checking is to be printed during learning.
    :return bool: True if absolute error for all properties is smaller then property_based_stopping[2].
    """
    from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

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


def bisimilar(a1: DeterministicAutomaton, a2: DeterministicAutomaton, return_cex: bool = False) -> bool | None | tuple:
    """
    Checks whether the provided automata are bisimilar.
    If return_cex the function returns a counter example or None, otherwise a Boolean is returned.

    :param DeterministicAutomaton a1: First automaton.
    :param DeterministicAutomaton a2: Second automaton.
    :param bool return_cex: Whether to return a counterexample instead of a boolean.
    :return bool | None | tuple: True or false if return_cex is set to False, otherwise None (no counterexample) or
        a counterexample.
    """

    # TODO allow states as inputs instead of automata
    if a1.__class__ != a2.__class__:
        raise ValueError("tried to check bisimilarity of different automaton types")
    # if you use this function with the same object (does not make sense anyway)
    if a1 is a2:
        a2 = a1.copy()

    supported_automaton_types = (Dfa, MooreMachine, MealyMachine)
    if not isinstance(a1, supported_automaton_types):
        raise NotImplementedError(
            f"bisimilarity is not implemented for {a1.__class__.__name__}. Supported: {', '.join(t.__name__ for t in supported_automaton_types)}")

    to_check: Queue[tuple[AutomatonState, AutomatonState]] = Queue()
    to_check.put((a1.initial_state, a2.initial_state))
    requirements = dict()
    requirements[(a1.initial_state, a2.initial_state)] = ()

    while not to_check.empty():
        s1, s2 = to_check.get()

        # check output equivalence for Dfa / Moore
        if isinstance(s1, (DfaState, MooreState)) and s1.output != s2.output:
            return requirements[(s1, s2)] if return_cex else False

        # check whether the same inputs are enabled + output equivalence for Mealy
        t1, t2 = s1.transitions, s2.transitions
        for t in it.chain(t1.keys(), filter(lambda x : x not in t1.keys(), t2.keys())):
            common = t in t1.keys() and t in t2.keys()
            if (not common) or (isinstance(s1, MealyState) and s1.output_fun[t] != s2.output_fun[t]) :
                return requirements[(s1, s2)] + (t,) if return_cex else False

        for t in t1.keys():
            c1, c2 = t1[t], t2[t]
            if (c1, c2) not in requirements:
                requirements[(c1, c2)] = requirements[(s1, s2)] + (t,)
                to_check.put((c1, c2))

    return None if return_cex else True


def compare_automata(aut_1: DeterministicAutomaton, aut_2: DeterministicAutomaton, num_cex: int = 10) -> list:
    """
    Finds cases of non-conformance between first and second automaton. This is done by performing RandomW equivalence
    check. It is possible that number of found counterexamples is smaller than num_cex, as no counterexample will be a
    suffix of a previously found counterexample.

    :param DeterministicAutomaton aut_1: First automaton.
    :param DeterministicAutomaton aut_2: Second automaton.
    :param int num_cex: Max. number of searches for counterexamples.
    :return list: A list of input sequences that revel different behaviour on both automata. Counterexamples are
        sorted by length.
    """
    #
    from aalpy.oracles import RandomWMethodEqOracle
    # if you use this function with the same object (does not make sense anyway)
    if aut_1 is aut_2:
        aut_2 = aut_1.copy()

    if set(aut_1.get_input_alphabet()) != set(aut_2.get_input_alphabet()):
        assert False, "Cannot compare automata with different input alphabets"

    input_al = aut_1.get_input_alphabet()
    # larger automaton is used as hypothesis, as then test-cases will contain prefixes leading to states
    # not in smaller automaton
    base_automaton, test_automaton = (aut_1, aut_2) if aut_1.size < aut_2.size else (aut_2, aut_1)
    base_sul = AutomatonSUL(base_automaton)

    # compute prefixes for all states of the test automaton (needed for advanced eq. oracle)
    for state in test_automaton.states:
        if not state.prefix:
            state.prefix = test_automaton.get_shortest_path(test_automaton.initial_state, state)

    # setup  the eq oracle
    eq_oracle = RandomWMethodEqOracle(input_al, base_sul, walks_per_state=min(100, len(input_al) * 10), walk_len=10)

    found_cex = []
    # to avoid near "infinite" loops due to while loop and set requirement
    # that is, if you can only find 1 cex and all other cexs are suffixes of that cex, first while condition will never
    # be reached
    failsafe_counter = 0
    failsafe_stopping = num_cex * 100
    while len(found_cex) < num_cex and failsafe_counter < failsafe_stopping:
        cex = eq_oracle.find_cex(test_automaton)
        # if no counterexample can be found terminate the loop
        if cex is None:
            break
        if cex not in found_cex:
            found_cex.append(cex)
        failsafe_counter += 1

    found_cex.sort(key=len)

    return found_cex


class TestCaseWrapperSUL(SUL):
    """
    SUL wrapper that records every membership query as a test case (input sequence, output sequence).
    """

    def __init__(self, sul: SUL) -> None:
        """
        Creates a test-case-recording wrapper around a SUL.

        :param SUL sul: The wrapped system under learning.
        """
        super().__init__()
        self.sul = sul
        self.test_cases = []
        self.test_case_inputs = None
        self.test_case_outputs = None

    def pre(self) -> None:
        """
        Resets the recorded input/output buffers and the wrapped SUL.
        """
        self.test_case_inputs = []
        self.test_case_outputs = []
        return self.sul.pre()

    def post(self) -> None:
        """
        Stores the recorded query as a test case and performs cleanup on the wrapped SUL.
        """
        if self.test_case_inputs and self.test_case_outputs:
            self.test_cases.append((tuple(self.test_case_inputs), tuple(self.test_case_outputs)))
        return self.sul.post()

    def step(self, letter: Any) -> Any:
        """
        Executes an action on the wrapped SUL and records the input/output pair.

        :param Any letter: Single input that is executed on the SUL.
        :return Any: Output received after executing the input.
        """
        output = self.sul.step(letter)
        self.test_case_inputs.append(letter)
        self.test_case_outputs.append(output)
        return output


def generate_test_cases(automaton: DeterministicAutomaton, oracle: Any) -> list:
    """
    Uses parametrized eq. oracle to construct test cases on the automaton.
    If automaton are big (200+ states), increase recursion depth if necessary (eg. sys.setrecursionlimit(10000)).

    :param DeterministicAutomaton automaton: Deterministic automaton that serves as a basis for test case generation.
    :param Any oracle: Oracle that will construct test-cases and record inputs and outputs.
    :return list: List of test cases, where each testcase is a tuple containing two elements, and input and an
        output sequance.
    """
    from copy import deepcopy

    automaton_copy = deepcopy(automaton)
    base_sul = AutomatonSUL(automaton_copy)

    wrapped_sul = TestCaseWrapperSUL(base_sul)
    oracle.sul = wrapped_sul
    # no counterexamples can be found
    cex = oracle.find_cex(automaton)
    assert cex is None
    return wrapped_sul.test_cases


def statistical_model_checking(model: Any, goals: set, max_num_steps: int, num_tests: int = 105967) -> float:
    """
    Estimates the probability of reaching a goal output within a bounded number of steps by sampling random tests.

    :param Any model: Model on which model checking is performed.
    :param set goals: Set of goal outputs.
    :param int max_num_steps: Bounded length of tests.
    :param int num_tests: Num of tests that will be performed.
    :return float: Num of tests containing element of goals set / num_tests.
    """

    def compute_output_sequence(model: Any, seq: list) -> set:
        """
        Executes an input sequence on the model from its initial state and collects the observed outputs.

        :param Any model: Model on which the sequence is executed.
        :param list seq: Input sequence to execute.
        :return set: Set of observed outputs.
        """
        model.reset_to_initial()
        observed_outputs = {model.step(i) for i in seq}
        return observed_outputs

    goal_reached = 0
    inputs = model.get_input_alphabet()
    for _ in range(num_tests):
        test_sequence = choices(inputs, k=max_num_steps)
        outputs = compute_output_sequence(model, test_sequence)
        if goals & outputs:
            goal_reached += 1

    return goal_reached / num_tests
