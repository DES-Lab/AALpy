import re
import sys
import traceback
from pathlib import Path

from pydot import Dot, Node, Edge

from aalpy.automata import Dfa, MooreMachine, Mdp, Onfsm, MealyState, DfaState, MooreState, MealyMachine, \
    MdpState, StochasticMealyMachine, StochasticMealyState, OnfsmState, MarkovChain, McState, Sevpa, SevpaState, \
    SevpaTransition, Vpa, VpaState, VpaTransition, NDMooreMachine, NDMooreState

file_types = ['dot', 'png', 'svg', 'pdf', 'string']
automaton_types = {Dfa: 'dfa', MealyMachine: 'mealy', MooreMachine: 'moore', Mdp: 'mdp',
                   StochasticMealyMachine: 'smm', Onfsm: 'onfsm', NDMooreMachine: 'ndmoore', MarkovChain: 'mc',
                   Sevpa: 'sevpa', Vpa: 'vpa'}


def _wrap_label(label):
    """
    Adds a " " around a label if not already present on both ends.
    """
    if label[0] == '\"' and label[-1] == '\"':
        return label
    return f'\"{label}\"'


def _get_node(state, automaton_type):
    if automaton_type == 'dfa':
        if state.is_accepting:
            return Node(state.state_id, label=_wrap_label(state.state_id), shape='doublecircle')
        return Node(state.state_id, label=_wrap_label(state.state_id))
    if automaton_type == 'mealy':
        return Node(state.state_id, label=_wrap_label(state.state_id))
    if automaton_type in ['moore', 'ndmoore']:
        return Node(state.state_id, label=_wrap_label(f'{state.state_id}|{state.output}'), shape='record',
                    style='rounded')
    if automaton_type == 'onfsm':
        return Node(state.state_id, label=_wrap_label(state.state_id))
    if automaton_type == 'mc':
        return Node(state.state_id, label=_wrap_label(f'{state.output}'))
    if automaton_type == 'mdp':
        return Node(state.state_id, label=_wrap_label(f'{state.output}'))
    if automaton_type == 'smm':
        return Node(state.state_id, label=_wrap_label(state.state_id))
    if automaton_type == 'sevpa' or automaton_type == 'vpa':
        if state.is_accepting:
            return Node(state.state_id, label=_wrap_label(state.state_id), shape='doublecircle')
        return Node(state.state_id, label=_wrap_label(state.state_id))


def _add_transition_to_graph(graph, state, automaton_type, display_same_state_trans, round_floats):
    if automaton_type == 'dfa' or automaton_type == 'moore':
        for i in state.transitions.keys():
            new_state = state.transitions[i]
            if not display_same_state_trans and new_state.state_id == state.state_id:
                continue
            graph.add_edge(Edge(state.state_id, new_state.state_id, label=_wrap_label(f'{i}')))
    if automaton_type == 'mealy':
        for i in state.transitions.keys():
            new_state = state.transitions[i]
            if not display_same_state_trans and new_state.state_id == state.state_id:
                continue
            graph.add_edge(Edge(state.state_id, new_state.state_id, label=_wrap_label(f'{i}/{state.output_fun[i]}')))
    if automaton_type == 'onfsm':
        for i in state.transitions.keys():
            new_state = state.transitions[i]
            for s in new_state:
                if not display_same_state_trans and state.state_id == s[1].state_id:
                    continue
                graph.add_edge(Edge(state.state_id, s[1].state_id, label=_wrap_label(f'{i}/{s[0]}')))
    if automaton_type == 'ndmoore':
        for i in state.transitions.keys():
            new_states = state.transitions[i]
            for new_state in new_states:
                if not display_same_state_trans and state.state_id == new_state.state_id:
                    continue
                graph.add_edge(Edge(state.state_id, new_state.state_id, label=_wrap_label(f'{i}')))
    if automaton_type == 'mc':
        for new_state, prob in state.transitions:
            prob = round(prob, round_floats) if round_floats else prob
            graph.add_edge(Edge(state.state_id, new_state.state_id, label=f'{prob}'))
    if automaton_type == 'mdp':
        for i in state.transitions.keys():
            new_state = state.transitions[i]
            for s in new_state:
                if not display_same_state_trans and s[0].state_id == state.state_id:
                    continue
                prob = round(s[1], round_floats) if round_floats else s[1]
                graph.add_edge(Edge(state.state_id, s[0].state_id, label=_wrap_label(f'{i}:{prob}')))
    if automaton_type == 'smm':
        for i in state.transitions.keys():
            new_state = state.transitions[i]
            for s in new_state:
                if not display_same_state_trans and s[0].state_id == state.state_id:
                    continue
                prob = round(s[2], round_floats) if round_floats else s[2]
                graph.add_edge(Edge(state.state_id, s[0].state_id, label=_wrap_label(f'{i}/{s[1]}:{prob}')))
    if automaton_type == 'sevpa':
        for i in state.transitions.keys():
            transitions_list = state.transitions[i]
            for transition in transitions_list:
                if transition.action == 'pop':
                    edge = Edge(state.state_id, transition.target_state.state_id,
                                label=_wrap_label(f'{transition.letter} / {transition.stack_guard}'))
                elif transition.action is None:
                    edge = Edge(state.state_id, transition.target_state.state_id,
                                label=_wrap_label(f'{transition.letter}'))
                else:
                    assert False
                graph.add_edge(edge)
    if automaton_type == 'vpa':
        for i in state.transitions.keys():
            transitions_list = state.transitions[i]
            for transition in transitions_list:
                if transition.action == 'pop':
                    edge = Edge(state.state_id, transition.target_state.state_id,
                                label=_wrap_label(f'{transition.letter} / pop({transition.stack_guard})'))
                elif transition.action == 'push':
                    edge = Edge(state.state_id, transition.target_state.state_id,
                                label=_wrap_label(f'{transition.letter} / push({transition.stack_guard})'))
                elif transition.action is None:
                    edge = Edge(state.state_id, transition.target_state.state_id,
                                label=_wrap_label(f'{transition.letter}'))
                else:
                    assert False
                graph.add_edge(edge)


def visualize_automaton(automaton, path="LearnedModel", file_type="pdf", display_same_state_trans=True):
    """
    Create a graphical representation of the automaton.
    Function is round in the separate thread in the background.
    If possible, it will be opened by systems default program.

    Args:

        automaton: automaton to be visualized

        path: pathlike or str, file in which visualization will be saved (Default value = "LearnedModel.pdf")

        file_type: type of file/visualization. Can be ['png', 'svg', 'pdf'] (Default value = "pdf")

        display_same_state_trans: if True, same state transitions will be displayed (Default value = True)

    """
    print('Visualization started in the background thread.')

    if len(automaton.states) >= 25:
        print(f'Visualizing {len(automaton.states)} state automaton could take some time.')

    import threading
    visualization_thread = threading.Thread(target=save_automaton_to_file, name="Visualization",
                                            args=(automaton, path, file_type, display_same_state_trans, True, 2))
    visualization_thread.start()


def save_automaton_to_file(automaton, path="LearnedModel", file_type="dot",
                           display_same_state_trans=True, visualize=False, round_floats=None):
    """
    The Standard of the automata strictly follows the syntax found at: https://automata.cs.ru.nl/Syntax/Overview.
    For non-deterministic and stochastic systems syntax can be found on AALpy's Wiki.

    Args:

        automaton: automaton to be saved to file

        path: pathlike or str, file in which visualization will be saved (Default value = "LearnedModel")

        file_type: type of file/visualization. Can be ['dot', 'png', 'svg', 'pdf'] (Default value = "dot)

        display_same_state_trans: True, should not be set to false except from the visualization method
            (Default value = True)

        visualize: visualize the automaton

        round_floats: for stochastic automata, round the floating point numbers to defined number of decimal places

    Returns:

    """
    path = Path(path)
    file_type = file_type.lower()
    assert file_type in file_types, f"Filetype {file_type} not in allowed filetypes"
    path = path.with_suffix(f".{file_type}")
    if file_type == 'dot' and not display_same_state_trans:
        print("When saving to file all transitions will be saved")
        display_same_state_trans = True
    automaton_type = automaton_types[automaton.__class__]

    graph = Dot(path.stem, graph_type='digraph')
    for state in automaton.states:
        if automaton_type in {'vpa', 'sevpa'} and state.state_id == 'ErrorSinkState':
            continue
        graph.add_node(_get_node(state, automaton_type))

    for state in automaton.states:
        _add_transition_to_graph(graph, state, automaton_type, display_same_state_trans, round_floats)

    # add initial node
    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', automaton.initial_state.state_id, label=''))

    if file_type == 'string':
        return graph.to_string()
    else:
        try:
            graph.write(path=path, format=file_type if file_type != 'dot' else 'raw')
            print(f'Model saved to {path.as_posix()}.')

            if visualize and file_type in {'pdf', 'png', 'svg'}:
                try:
                    import webbrowser
                    webbrowser.open(path.resolve().as_uri())
                except OSError:
                    traceback.print_exc()
                    print(f'Could not open the file {path.as_posix()}.', file=sys.stderr)
        except OSError:
            traceback.print_exc()
            print(f'Could not write to the file {path.as_posix()}.', file=sys.stderr)


sevpa_transition_pattern = r"(\S+)\s*/\s*\(\s*'(\S+)'\s*,\s*'(\S+)'\s*\)"
vpa_push_pattern = r"(\S+)\s*/\s*push\(\s*(.*?)\s*\)"
vpa_pop_pattern = r"(\S+)\s*/\s*pop\(\s*(.*?)\s*\)"


def _process_label(label, source, destination, automaton_type):
    if automaton_type == 'dfa' or automaton_type == 'moore':
        source.transitions[int(label) if label.isdigit() else label] = destination
    if automaton_type == 'mealy':
        inp, out = label.split('/', maxsplit=1)
        inp = int(inp) if inp.isdigit() else inp
        out = int(out) if out.isdigit() else out
        source.transitions[inp] = destination
        source.output_fun[inp] = out
    if automaton_type == 'onfsm':
        inp, out = label.split('/', maxsplit=1)
        inp = int(inp) if inp.isdigit() else inp
        out = int(out) if out.isdigit() else out
        source.transitions[inp].append((out, destination))
    if automaton_type == 'ndmoore':
        source.transitions[int(label) if label.isdigit() else label].append(destination)
    if automaton_type == 'mc':
        prob = label
        source.transitions.append((destination, float(prob)))
    if automaton_type == 'mdp':
        inp, prob = label.split(':', maxsplit=1)
        inp = int(inp) if inp.isdigit() else inp
        prob = float(prob)
        source.transitions[inp].append((destination, prob))
    if automaton_type == 'smm':
        inp, out_prob = label.split('/', maxsplit=1)
        out, prob = out_prob.split(':', maxsplit=1)
        inp = int(inp) if inp.isdigit() else inp
        out = int(out) if out.isdigit() else out
        source.transitions[inp].append((destination, out, float(prob)))
    if automaton_type == 'sevpa':
        match = re.match(sevpa_transition_pattern, label)
        # cast to integer
        label = int(label) if label.isdigit() else label
        if match:
            ret, stack_guard, top_of_stack = match.groups()
            return_transition = SevpaTransition(destination, ret, 'pop', (stack_guard, top_of_stack))
            source.transitions[label].append(return_transition)
        else:
            internal_transition = SevpaTransition(destination, label, None, None)
            source.transitions[label].append(internal_transition)
        pass
    if automaton_type == 'vpa':
        input_symbol, stack_symbol, action = None, None, None
        if 'push' in label:
            push_elements = re.match(vpa_push_pattern, label)
            input_symbol, stack_symbol = push_elements.group(1), push_elements.group(2)
            action = 'push'
        elif 'pop' in label:
            pop_elements = re.match(vpa_pop_pattern, label)
            input_symbol, stack_symbol = pop_elements.group(1), pop_elements.group(2)
            action = 'pop'
        else:
            input_symbol = label

        input_symbol = int(input_symbol) if input_symbol.isdigit() else input_symbol
        if stack_symbol:
            stack_symbol = int(stack_symbol) if stack_symbol.isdigit() else stack_symbol

        transition = VpaTransition(source, destination, input_symbol, action, stack_symbol)

        source.transitions[input_symbol].append(transition)


def _process_node_label(node, label, node_label_dict, node_type, automaton_type):
    node_name = node.get_name()
    if automaton_type == 'mdp' or automaton_type == 'mc':
        node_label_dict[node_name] = node_type(node_name, label)
    else:
        if automaton_type == 'moore' and label != "":
            label_output = _strip_label(label)
            label, output = label_output.split("|", maxsplit=1)
            output = output.strip() if not output.isdigit() else int(output)
            node_label_dict[node_name] = node_type(label, output)
        else:
            node_label_dict[node_name] = node_type(label)
        if automaton_type == 'dfa' or automaton_type == 'sevpa':
            if 'shape' in node.get_attributes().keys() and 'doublecircle' in node.get_attributes()['shape']:
                node_label_dict[node_name].is_accepting = True


def _strip_label(label: str) -> str:
    label = label.strip()
    if label[0] == '\"' and label[-1] == label[0]:
        label = label[1:-1]
    if label[0] == '{' and label[-1] == '}':
        label = label[1:-1]
    # label = label.replace(" ", "")
    return label


def _process_node_label_prime(node_name, label, line, node_label_dict, node_type, automaton_type):
    if automaton_type == 'mdp' or automaton_type == 'mc':
        node_label_dict[node_name] = node_type(node_name, label)
    else:
        if automaton_type in {'moore', 'ndmoore'} and label != "":
            label_output = _strip_label(label)
            if "|" in label_output:
                label, output = label_output.split("|", maxsplit=1)
            else:
                label = node_name
                output = label_output
            output = output.strip() if not output.isdigit() else int(output)
            node_label_dict[node_name] = node_type(label, output)
        else:
            node_label_dict[node_name] = node_type(label)
        if automaton_type in {'dfa', 'vpa', 'sevpa'}:
            if 'doublecircle' in line:
                node_label_dict[node_name].is_accepting = True


# TODO: robust patterns (break eg. if state label contains "-")
label_pattern = r'label=("[^"]*"|[^\s\],]*)'
starting_state_pattern = r'__start0\s*->\s*(\w+)\s*(?:\[label=""\])?;?'
transition_pattern = r'(\w+)\s*->\s*(\w+)\s*(.*)?;?'


def load_automaton_from_file(path, automaton_type, compute_prefixes=False):
    """
    Loads the automaton from the file.
    Standard of the automatas strictly follows syntax found at: https://automata.cs.ru.nl/Syntax/Overview.
    For non-deterministic and stochastic systems syntax can be found on AALpy's Wiki.

    Args:

        path: pathlike or str to the file

        automaton_type: type of the automaton, one of ['dfa', 'mealy', 'moore', 'mdp', 'smm',
                        'onfsm', 'ndmoore', 'mc', 'sevpa', 'vpa']

        compute_prefixes: it True, shortest path to reach every state will be computed and saved in the prefix of
            the state. Useful when loading the model to use them as a equivalence oracle. (Default value = False)

    Returns:

      loaded automaton

    """
    assert automaton_type in automaton_types.values()

    id_node_aut_map = {'dfa': (DfaState, Dfa), 'mealy': (MealyState, MealyMachine), 'moore': (MooreState, MooreMachine),
                       'onfsm': (OnfsmState, Onfsm), 'mdp': (MdpState, Mdp), 'mc': (McState, MarkovChain),
                       'smm': (StochasticMealyState, StochasticMealyMachine), 'sevpa': (SevpaState, Sevpa),
                       'vpa': (VpaState, Vpa), 'ndmoore': (NDMooreState, NDMooreMachine)}

    nodeType, aut_type = id_node_aut_map[automaton_type]

    initial_state = None
    transition_data = []

    node_label_dict = dict()

    with open(Path(path)) as f:
        for line in f.readlines():
            line = line.strip()
            if '__start0 ->' in line:
                match = re.search(starting_state_pattern, line)
                initial_state = match.group(1).strip()
            # State definitions
            elif '__start0' not in line and 'label' in line and '->' not in line:
                state_id = line.split('[')[0].strip()
                match = re.search(label_pattern, line)
                label = _strip_label(match.group(1))
                _process_node_label_prime(state_id, label, line, node_label_dict, nodeType, automaton_type)
            # transitions
            elif '->' in line:
                match = re.match(transition_pattern, line)
                # source, destination, label
                label = re.search(label_pattern, match.group(3)).group(1)
                transition_data.append((match.group(1).strip(), match.group(2).strip(), label))

    # ensure initial state is defined and it is defined states
    assert initial_state is not None and initial_state in node_label_dict.keys()
    initial_state = node_label_dict[initial_state]

    for source, destination, transition_label in transition_data:
        label = _strip_label(transition_label)
        _process_label(label, node_label_dict[source], node_label_dict[destination], automaton_type)

    automaton = aut_type(initial_state, list(node_label_dict.values()))
    if automaton_type not in {'mc', 'sevpa', 'vpa'} and not automaton.is_input_complete():
        print('Warning: Loaded automaton is not input complete.')
    if compute_prefixes and not automaton_type not in {'mc', 'sevpa', 'vpa'}:
        for state in automaton.states:
            state.prefix = automaton.get_shortest_path(automaton.initial_state, state)
    return automaton
