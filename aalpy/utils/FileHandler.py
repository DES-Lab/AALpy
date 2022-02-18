import os
import sys
import traceback

from pydot import Dot, Node, Edge, graph_from_dot_file

from aalpy.automata import Dfa, MooreMachine, Mdp, Onfsm, MealyState, DfaState, MooreState, MealyMachine, \
    MdpState, StochasticMealyMachine, StochasticMealyState, OnfsmState, MarkovChain, McState

file_types = ['dot', 'png', 'svg', 'pdf', 'string']
automaton_types = ['dfa', 'mealy', 'moore', 'mdp', 'smm', 'onfsm', 'mc']


def visualize_automaton(automaton, path="LearnedModel", file_type='pdf', display_same_state_trans=True):
    """
    Create a graphical representation of the automaton.
    Function is round in the separate thread in the background.
    If possible, it will be opened by systems default program.

    Args:

        automaton: automaton to be visualized

        path: file in which visualization will be saved (Default value = "Model_Visualization")

        file_type: type of file/visualization. Can be ['png', 'svg', 'pdf'] (Default value = 'pdf')

        display_same_state_trans: if True, same state transitions will be displayed (Default value = True)

    """
    print('Visualization started in the background thread.')

    if len(automaton.states) >= 25:
        print(f'Visualizing {len(automaton.states)} state automaton could take some time.')

    import threading
    visualization_thread = threading.Thread(target=save_automaton_to_file, name="Visualization",
                                            args=(automaton, path, file_type, display_same_state_trans, True, 2))
    visualization_thread.start()


def save_automaton_to_file(automaton, path="LearnedModel", file_type='dot',
                           display_same_state_trans=True, visualize=False, round_floats=None):
    """
    The Standard of the automata strictly follows the syntax found at: https://automata.cs.ru.nl/Syntax/Overview.
    For non-deterministic and stochastic systems syntax can be found on AALpy's Wiki.

    Args:

        automaton: automaton to be saved to file

        path: file in which automaton will be saved (Default value = "LearnedModel")

        file_type: Can be ['dot', 'png', 'svg', 'pdf'] (Default value = 'dot')

        display_same_state_trans: True, should not be set to false except from the visualization method
            (Default value = True)

        visualize: visualize the automaton

        round_floats: for stochastic automata, round the floating point numbers to defined number of decimal places

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
    is_mc = isinstance(automaton, MarkovChain)

    graph = Dot(path, graph_type='digraph')
    for state in automaton.states:
        if is_dfa and state.is_accepting:
            graph.add_node(Node(state.state_id, label=state.state_id, shape='doublecircle'))
        elif is_moore:
            graph.add_node(Node(state.state_id, label=f'{state.state_id}|{state.output}',
                                shape='record', style='rounded'))
        elif is_mdp or is_mc:
            graph.add_node(Node(state.state_id, label=f'{state.output}'))
        else:
            graph.add_node(Node(state.state_id, label=state.state_id))

    for state in automaton.states:
        if is_mc:
            for new_state, prob in state.transitions:
                prob = round(prob, round_floats) if round_floats else prob
                graph.add_edge(Edge(state.state_id, new_state.state_id, label=f'{prob}'))
            continue
        for i in state.transitions.keys():
            if isinstance(state, MealyState):
                new_state = state.transitions[i]
                if not display_same_state_trans and new_state.state_id == state.state_id:
                    continue
                graph.add_edge(Edge(state.state_id, new_state.state_id, label=f'{i}/{state.output_fun[i]}'))
            elif is_mdp:
                # here we do not have single state, but a list of (State, probability) tuples
                new_state = state.transitions[i]
                for s in new_state:
                    if not display_same_state_trans and s[0].state_id == state.state_id:
                        continue
                    prob = round(s[1], round_floats) if round_floats else s[1]
                    graph.add_edge(Edge(state.state_id, s[0].state_id, label=f'{i}:{prob}'))
            elif is_onsfm:
                new_state = state.transitions[i]
                for s in new_state:
                    if not display_same_state_trans and state.state_id == s[1].state_id:
                        continue
                    graph.add_edge(Edge(state.state_id, s[1].state_id, label=f'{i}/{s[0]}'))
            elif is_smm:
                new_state = state.transitions[i]
                for s in new_state:
                    if not display_same_state_trans and s[0].state_id == state.state_id:
                        continue
                    prob = round(s[2], round_floats) if round_floats else s[2]
                    graph.add_edge(Edge(state.state_id, s[0].state_id, label=f'{i}/{s[1]}:{prob}'))
            else:
                new_state = state.transitions[i]
                if not display_same_state_trans and new_state.state_id == state.state_id:
                    continue
                graph.add_edge(Edge(state.state_id, new_state.state_id, label=f'{i}'))

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', automaton.initial_state.state_id, label=''))

    if file_type == 'string':
        return graph.to_string()
    else:
        try:
            graph.write(path=f'{path}.{file_type}', format=file_type if file_type != 'dot' else 'raw')
            print(f'Model saved to {path}.{file_type}.')

            if visualize and file_type in {'pdf', 'png', 'svg'}:
                try:
                    import webbrowser
                    abs_path = os.path.abspath(f'{path}.{file_type}')
                    path = f'file:///{abs_path}'
                    webbrowser.open(path)
                except OSError:
                    traceback.print_exc()
                    print(f'Could not open the file {path}.{file_type}.', file=sys.stderr)
        except OSError:
            traceback.print_exc()
            print(f'Could not write to the file {path}.{file_type}.', file=sys.stderr)


def load_automaton_from_file(path, automaton_type, compute_prefixes=False):
    """
    Loads the automaton from the file.
    Standard of the automatas strictly follows syntax found at: https://automata.cs.ru.nl/Syntax/Overview.
    For non-deterministic and stochastic systems syntax can be found on AALpy's Wiki.

    Args:

        path: path to the file

        automaton_type: type of the automaton, if not specified it will be automatically determined according,
            one of ['dfa', 'mealy', 'moore', 'mdp', 'smm', 'onfsm', 'mc']

        compute_prefixes: it True, shortest path to reach every state will be computed and saved in the prefix of
            the state. Useful when loading the model to use them as a equivalence oracle. (Default value = False)

    Returns:
      automaton

    """
    graph = graph_from_dot_file(path)[0]

    assert automaton_type in automaton_types

    id_node_aut_map = {'dfa': (DfaState, Dfa), 'mealy': (MealyState, MealyMachine), 'moore': (MooreState, MooreMachine),
                       'onfsm': (OnfsmState, Onfsm), 'mdp': (MdpState, Mdp), 'mc': (McState, MarkovChain),
                       'smm': (StochasticMealyState, StochasticMealyMachine)}

    node, aut_type = id_node_aut_map[automaton_type]

    node_label_dict = dict()
    for n in graph.get_node_list():
        if n.get_name() == '__start0' or n.get_name() == '' or n.get_name() == '"\\n"':
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
            output = int(output) if output.isdigit() else output

        if automaton_type == 'mdp' or automaton_type == 'mc':
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
        elif automaton_type == 'mc':
            prob = label
            source.transitions.append((destination, float(prob)))
        else:  # moore or dfa
            source.transitions[int(label) if label.isdigit() else label] = destination

    if initial_node is None:
        print("No initial state found. \n"
              "Please follow syntax found at: https://github.com/DES-Lab/AALpy/wiki/"
              "Loading,Saving,-Syntax-and-Visualization-of-Automata ")
        assert False

    automaton = aut_type(initial_node, list(node_label_dict.values()))
    if automaton_type != 'mc' and not automaton.is_input_complete():
        print('Warning: Loaded automaton is not input complete.')
    if compute_prefixes and not automaton_type == 'mc':
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


def visualize_fpta(red):
    red_sorted = sorted(list(red), key=lambda x: len(x.prefix))
    graph = Dot('fpta', graph_type='digraph')

    for i, r in enumerate(red_sorted):
        r.state_id = f'q{i}'
        graph.add_node(Node(r.state_id, label=r.state_id))

    for r in red_sorted:
        for i, c in r.children.items():
            graph.add_edge(Edge(r.state_id, c.state_id, label=i))

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', red_sorted[0].state_id, label=''))

    return graph
