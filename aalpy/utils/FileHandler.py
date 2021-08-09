from aalpy.base import Automaton
import os

from pydot import Dot, Node, Edge, graph_from_dot_file

from aalpy.automata import Dfa, MooreMachine, Mdp, Onfsm, MealyState, DfaState, MooreState, MealyMachine, \
    MdpState, StochasticMealyMachine, StochasticMealyState, OnfsmState, IotsState, IotsMachine

file_types = ['dot', 'png', 'svg', 'pdf', 'string']


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
        print(
            f'Visualizing {len(automaton.states)} state automaton could take some time.')

    import threading
    visualization_thread = threading.Thread(target=save_automaton_to_file, name="Visualization",
                                            args={automaton: automaton, path: path, file_type: file_type,
                                                  display_same_state_trans: display_same_state_trans})
    visualization_thread.start()


def save_automaton_to_file(automaton, path="LearnedModel", file_type='dot',
                           display_same_state_trans=True):
    """
    The Standard of the automata strictly follows the syntax found at: https://automata.cs.ru.nl/Syntax/Overview.
    For non-deterministic and stochastic systems syntax can be found on AALpy's Wiki.

    Args:

        automaton: automaton to be saved to file

        path: file in which automaton will be saved (Default value = "LearnedModel")

        file_type: Can be ['dot', 'png', 'svg', 'pdf'] (Default value = 'dot')

        display_same_state_trans: True, should not be set to false except from the visualization method
            (Default value = True)

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

    graph = Dot(path, graph_type='digraph')
    for state in automaton.states:
        if is_dfa and state.is_accepting:
            graph.add_node(
                Node(state.state_id, label=state.state_id, shape='doublecircle'))
        elif is_moore:
            graph.add_node(Node(state.state_id, label=f'{state.state_id}|{state.output}',
                                shape='record', style='rounded'))
        elif is_mdp:
            graph.add_node(Node(state.state_id, label=f'{state.output}'))
        else:
            graph.add_node(Node(state.state_id, label=state.state_id))

    for state in automaton.states:
        for i in state.transitions.keys():
            if isinstance(state, MealyState):
                new_state = state.transitions[i]
                if not display_same_state_trans and new_state.state_id == state.state_id:
                    continue
                graph.add_edge(
                    Edge(state.state_id, new_state.state_id, label=f'{i}/{state.output_fun[i]}'))
            elif is_mdp:
                # here we do not have single state, but a list of (State, probability) tuples
                new_state = state.transitions[i]
                for s in new_state:
                    if not display_same_state_trans and s[0].state_id == state.state_id:
                        continue
                    graph.add_edge(
                        Edge(state.state_id, s[0].state_id, label=f'{i} : {round(s[1], 2)}'))
            elif is_onsfm:
                new_state = state.transitions[i]
                for s in new_state:
                    if not display_same_state_trans and state.state_id == s[1].state_id:
                        continue
                    graph.add_edge(
                        Edge(state.state_id, s[1].state_id, label=f'{i}/{s[0]}'))
            elif is_smm:
                new_state = state.transitions[i]
                for s in new_state:
                    if not display_same_state_trans and s[0].state_id == state.state_id:
                        continue
                    graph.add_edge(
                        Edge(state.state_id, s[0].state_id, label=f'{i}/{s[1]}:{round(s[2], 2)}'))
            else:
                new_state = state.transitions[i]
                if not display_same_state_trans and new_state.state_id == state.state_id:
                    continue
                graph.add_edge(
                    Edge(state.state_id, new_state.state_id, label=f'{i}'))

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(
        Edge('__start0', automaton.initial_state.state_id, label=''))

    if file_type == 'string':
        return graph.to_string()
    elif file_type == 'dot':
        graph.write(path=f'{path}.dot', format='raw')
    else:
        try:
            graph.write(path=f'{path}.{file_type}', format=file_type)
            print(f'Visualized model saved to {path}.{file_type}.')

            try:
                import webbrowser
                abs_path = os.path.abspath(f'{path}.{file_type}')
                path = f'file:///{abs_path}'
                webbrowser.open(path)
            except OSError:
                pass
        except OSError:
            print(f'Could not write to file {path}.{file_type} (Permission denied).'
                  f'If the file is open, close it and retry.')


def load_automaton_from_file(path, automaton_type, compute_prefixes=False):
    """
    Loads the automaton from the file.
    Standard of the automatas strictly follows syntax found at: https://automata.cs.ru.nl/Syntax/Overview.
    For non-deterministic and stochastic systems syntax can be found on AALpy's Wiki.

    Args:

        path: path to the file

        automaton_type: type of the automaton, if not specified it will be automatically determined according,
            one of ['dfa', 'mealy', 'moore', 'mdp', 'smm', 'onfsm', iots]

        compute_prefixes: it True, shortest path to reach every state will be computed and saved in the prefix of
            the state. Useful when loading the model to use them as a equivalence oracle. (Default value = False)

    Returns:
      automaton

    """
    graph = graph_from_dot_file(path)[0]

    def get_class_information(automaton_type):
        if automaton_type == 'dfa':
            return (DfaState, Dfa)
        elif automaton_type == 'mealy':
            return (MealyState, MealyMachine)
        elif automaton_type == 'moore':
            return (MooreState, MooreMachine)
        elif automaton_type == 'mdp':
            return (MdpState, Mdp)
        elif automaton_type == 'smm':
            return (StochasticMealyState, StochasticMealyMachine)
        elif automaton_type == 'onfsm':
            return (OnfsmState, Onfsm)
        elif automaton_type == 'iots':
            return (IotsState, IotsMachine)
        else:
            assert False, "Automaton type is unknown"

    def create_state_label_dict(graph: list, node_class):
        result = dict()

        for n in graph.get_node_list():
            node_name = None
            label = None
            output = None

            node_name = n.get_name()
            if node_name == '__start0' or node_name == '':
                continue

            if 'label' in n.get_attributes().keys():
                label = _process_label(n.get_attributes()['label'])

            if node_class == MdpState:
                result[node_name] = node_class(node_name, label)
            elif node_class == MooreState:
                label_output = _process_label(label)
                label = label_output.split('|')[0]
                output = label_output.split('|')[1]
                output = int(output) if output.isdigit() else output
                result[node_name] = node_class(label, output)
            elif node_class == DfaState:
                is_accepting_state = 'shape' in n.get_attributes().keys(
                ) and 'doublecircle' in n.get_attributes()['shape']
                result[node_name] = node_class(label, is_accepting_state)
            else:
                result[node_name] = node_class(label)

        return result

    def get_initial_state(graph, state_label_dict: dict):
        for edge in graph.get_edge_list():
            if edge.get_source() == '__start0':
                return state_label_dict[edge.get_destination()]

        print("No initial state found. \n"
              "Please follow syntax found at: https://github.com/DES-Lab/AALpy/wiki/"
              "Loading,Saving,-Syntax-and-Visualization-of-Automata ")
        assert False

    def update_states(graph, state_label_dict: dict):
        for edge in graph.get_edge_list():

            if edge.get_source() == '__start0':
                continue

            source = state_label_dict[edge.get_source()]
            destination = state_label_dict[edge.get_destination()]
            label = _process_label(edge.get_attributes()['label'])

            if isinstance(source, MealyState):
                inp = label.split('/')[0]
                out = label.split('/')[1]
                inp = int(inp) if inp.isdigit() else inp
                out = int(out) if out.isdigit() else out
                source.transitions[inp] = destination
                source.output_fun[inp] = out
            elif isinstance(source, OnfsmState):
                inp = label.split('/')[0]
                out = label.split('/')[1]
                inp = int(inp) if inp.isdigit() else inp
                out = int(out) if out.isdigit() else out
                source.transitions[inp].append((out, destination))
            elif isinstance(source, StochasticMealyState):
                inp = label.split('/')[0]
                out_prob = label.split('/')[1]
                out = out_prob.split(':')[0]
                prob = out_prob.split(':')[1]
                inp = int(inp) if inp.isdigit() else inp
                out = int(out) if out.isdigit() else out
                source.transitions[inp].append((destination, out, float(prob)))
            elif isinstance(source, MdpState):
                inp = label.split(':')[0]
                prob = label.split(':')[1]
                inp = int(inp) if inp.isdigit() else inp
                prob = float(prob)
                source.transitions[inp].append((destination, prob))
            elif isinstance(source, IotsState):
                if label.startswith('?'):
                    source.add_input(label, destination)
                elif label.startswith('!'):
                    source.add_output(label, destination)
                else:
                    assert False, "No prefix found."
            else:
                label = int(label) if label.isdigit() else label
                source.transitions[label] = destination

    def build_automaton(automaton_class, state_label_dict, initial_state, compute_prefixes: bool):
        states = list(state_label_dict.values())
        automaton: Automaton = automaton_class(initial_state, states)

        assert automaton.is_input_complete()

        if compute_prefixes:
            for state in automaton.states:
                state.prefix = automaton.get_shortest_path(
                    automaton.initial_state, state)

        return automaton

    (state_class, automaton_class) = get_class_information(automaton_type)
    state_label_dict = create_state_label_dict(graph, state_class)
    initial_state = get_initial_state(graph, state_label_dict)
    update_states(graph, state_label_dict)

    return build_automaton(
        automaton_class, state_label_dict, initial_state, compute_prefixes)


def _process_label(label: str) -> str:
    label = label.strip()
    if label[0] == '\"' and label[-1] == label[0]:
        label = label[1:-1]
    if label[0] == '{' and label[-1] == '}':
        label = label[1:-1]
    label = label.replace(" ", "")
    return label
