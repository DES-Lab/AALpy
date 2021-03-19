import os
from collections import defaultdict

from pydot import Dot, Node, Edge, graph_from_dot_file

from aalpy.automata import Dfa, MooreMachine, Mdp, Onfsm, MealyState, DfaState, MooreState, MealyMachine, \
    MdpState, StochasticMealyMachine, StochasticMealyState, OnfsmState

file_types = ['dot', 'png', 'svg', 'pdf', 'string']
automaton_types = ['dfa', 'mealy', 'moore', 'mdp', 'smm', 'onfsm']


def visualize_automaton(automaton, path="Model_Visualization", file_type='pdf', display_same_state_trans=True):
    """
    Create a graphical representation of the automaton.

    Args:

        automaton: automaton to be visualized

        path: file in which visualization will be saved (Default value = "Model_Visualization")

        file_type: type of file/visualization. Can be ['png', 'svg', 'pdf'] (Default value = 'pdf')

        display_same_state_trans: if True, same state transitions will be displayed (Default value = True)

    """
    save_automaton_to_file(automaton, path=path, file_type=file_type, display_same_state_trans=display_same_state_trans)


def save_automaton_to_file(automaton, path="LearnedModel", file_type='dot',
                           display_same_state_trans=True):
    """
    The Standard of the automata strictly follows the syntax found at: https://automata.cs.ru.nl/Syntax/Overview.

    Args:

        automaton: automaton to be saved to file

        path: file in which automaton will be saved (Default value = "LearnedModel")

        file_type: Can be ['dot', 'png', 'svg', 'pdf'] (Default value = 'dot')

        display_same_state_trans: True, should not be set to false except from the visualization method (Default value = True)

    Returns:

    """
    assert file_type in file_types
    if file_type == 'dot' and not display_same_state_trans:
        print("When saving to file all transitions will be saved")
        display_same_state_trans = True
    is_dfa = isinstance(automaton, Dfa)
    is_moore = isinstance(automaton, MooreMachine)
    is_mdp = isinstance(automaton, Mdp)
    is_onsfm = isinstance(automaton, Onfsm)
    is_smm = isinstance(automaton, StochasticMealyMachine)

    graph = Dot('path', graph_type='digraph')
    for state in automaton.states:
        if is_dfa and state.is_accepting:
            graph.add_node(Node(state.state_id, label=state.state_id, shape='doublecircle'))
        elif is_moore:
            graph.add_node(Node(state.state_id, label=f'{state.state_id}|{state.output}',
                                shape='record', style='rounded'))
        elif is_mdp:
            graph.add_node(Node(state.state_id, label=f'{state.output}'))
        else:
            graph.add_node(Node(state.state_id, label=state.state_id))

    for state in automaton.states:
        for i in state.transitions.keys():
            new_state = state.transitions[i]
            if not display_same_state_trans and state.state_id == new_state.state_id:
                continue
            if isinstance(state, MealyState):
                graph.add_edge(Edge(state.state_id, new_state.state_id, label=f'{i}/{state.output_fun[i]}'))
            elif is_mdp:
                # here we do not have single state, but a list of (State, probability) tuples
                for s in new_state:
                    graph.add_edge(Edge(state.state_id, s[0].state_id, label=f'{i} : {round(s[1], 2)}'))
            elif is_onsfm:
                for s in new_state:
                    graph.add_edge(Edge(state.state_id, s[1].state_id, label=f'{i}/{s[0]}'))
            elif is_smm:
                for s in new_state:
                    graph.add_edge(Edge(state.state_id, s[0].state_id, label=f'{i}/{s[1]}:{round(s[2], 2)}'))
            else:
                graph.add_edge(Edge(state.state_id, new_state.state_id, label=f'{i}'))

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', automaton.initial_state.state_id, label=''))

    if file_type == 'string':
        return graph.to_string()
    elif file_type == 'dot':
        graph.write(path=f'{path}.dot', format='raw')
    else:
        try:
            graph.write(path=f'{path}.{file_type}', format=file_type)
            print(f'Visualized model saved to {path}.{file_type}.')
        except OSError:
            print(f'Could not write to file {path}.{file_type} (Permission denied). Close the file if open and retry.')


def load_automaton_from_file(path, automaton_type=None, compute_prefixes=False):
    """
    Loads the automaton from the file.
    Standard of the automatas strictly follows syntax found at: https://automata.cs.ru.nl/Syntax/Overview.

    Args:

        path: path to the file

        automaton_type: type of the automaton, if not specified it will be automatically determined according (Default value = None)

        compute_prefixes: it True, shortest path to reach every state will be computed and saved in the prefix of
            the state. Useful when loading the model to use them as a equivalence oracle. (Default value = False)

    Returns:
      automaton

    """
    graph = graph_from_dot_file(path)[0]
    if not automaton_type:
        if 'doublecircle' in graph.to_string():
            automaton_type = 'dfa'
        elif 'record' in graph.to_string() and 'rounded' in graph.to_string():
            automaton_type = 'moore'
        else:
            automaton_type = 'mealy'
    assert automaton_type in automaton_types

    node = MealyState if automaton_type == 'mealy' else DfaState if automaton_type == 'dfa' else MooreState
    node = MdpState if automaton_type == 'mdp' else StochasticMealyState if automaton_type == 'smm' else node
    node = OnfsmState if automaton_type == 'onfsm' else node
    assert node is not None

    node_label_dict = dict()
    for n in graph.get_node_list():
        if n.get_name() == '__start0' or n.get_name() == '':
            continue
        label = None
        if 'label' in n.get_attributes().keys():
            label = n.get_attributes()['label']
            label = _process_label(label)

        node_name = n.get_name()
        output = None
        if automaton_type == 'moore' and label != "":
            label_output = _process_label(label)
            label = label_output.split('|')[0]
            output = label_output.split('|')[1]

        if automaton_type == 'mdp':
            node_label_dict[node_name] = node(node_name, label)
        else:
            node_label_dict[node_name] = node(label, output) if automaton_type == 'moore' else node(label)

        if 'shape' in n.get_attributes().keys() and 'doublecircle' in n.get_attributes()['shape']:
            node_label_dict[node_name].is_accepting = True

    initial_node = None
    for edge in graph.get_edge_list():
        if edge.get_source() == '__start0':
            initial_node = node_label_dict[edge.get_destination()]
            continue
        source = node_label_dict[edge.get_source()]
        destination = node_label_dict[edge.get_destination()]
        label = edge.get_attributes()['label']
        label = _process_label(label)
        if automaton_type == 'mealy':
            inp = label.split('/')[0]
            out = label.split('/')[1]
            inp = int(inp) if inp.isdigit() else inp
            out = int(out) if out.isdigit() else out
            source.transitions[inp] = destination
            source.output_fun[inp] = out
        elif automaton_type == 'onfsm':
            inp = label.split('/')[0]
            out = label.split('/')[1]
            inp = int(inp) if inp.isdigit() else inp
            out = int(out) if out.isdigit() else out
            source.transitions[inp].append((out, destination))
        elif automaton_type == 'smm':
            inp = label.split('/')[0]
            out_prob = label.split('/')[1]
            out = out_prob.split(':')[0]
            prob = out_prob.split(':')[1]
            inp = int(inp) if inp.isdigit() else inp
            out = int(out) if out.isdigit() else out
            source.transitions[inp].append((destination, out, float(prob)))
        elif automaton_type == 'mdp':
            inp = label.split(':')[0]
            prob = label.split(':')[1]
            inp = int(inp) if inp.isdigit() else inp
            prob = float(prob)
            source.transitions[inp].append((destination, prob))
        else:
            source.transitions[int(label) if label.isdigit() else label] = destination

    if automaton_type == 'dfa':
        automaton = Dfa(initial_node, list(node_label_dict.values()))
    elif automaton_type == 'moore':
        automaton = MooreMachine(initial_node, list(node_label_dict.values()))
    elif automaton_type == 'mdp':
        automaton = Mdp(initial_node, list(node_label_dict.values()))
    elif automaton_type == 'smm':
        automaton = StochasticMealyMachine(initial_node, list(node_label_dict.values()))
    elif automaton_type == 'onfsm':
        automaton = Onfsm(initial_node, list(node_label_dict.values()))
    else:
        automaton = MealyMachine(initial_node, list(node_label_dict.values()))
    assert automaton.is_input_complete()
    if compute_prefixes:
        for state in automaton.states:
            state.prefix = automaton.get_shortest_path(automaton.initial_state, state)
    return automaton


def _process_label(label: str) -> str:
    label = label.strip()
    if label[0] == '\"' and label[-1] == label[0]:
        label = label[1:-1]
    if label[0] == '{' and label[-1] == '}':
        label = label[1:-1]
    label = label.replace(" ", "")
    return label


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
