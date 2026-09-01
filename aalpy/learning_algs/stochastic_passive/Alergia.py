# Implementation of the Alergia/IOAlergia passive learning algorithm for Markov chains, MDPs, and stochastic Mealy
# machines, plus helper entry points for running it on in-memory data or via the external JAlergia implementation.
import time
from bisect import insort

from aalpy.automata import MarkovChain, MdpState, Mdp, McState, StochasticMealyState, \
    StochasticMealyMachine
from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import CompatibilityChecker, HoeffdingCompatibility
from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode, create_fpta

state_automaton_map = {'mc': (McState, MarkovChain), 'mdp': (MdpState, Mdp),
                       'smm': (StochasticMealyState, StochasticMealyMachine)}


class Alergia:
    """
    Implementation of the Alergia/IOAlergia state-merging algorithm, building an FPTA from data and merging
    statistically compatible states to obtain a Markov chain, MDP, or stochastic Mealy machine.
    """

    def __init__(self, data: list, automaton_type: str, eps: float | str = 0.05,
                 compatibility_checker: CompatibilityChecker | None = None, print_info: bool = False) -> None:
        """
        Creates an Alergia instance and constructs the FPTA from the provided data.

        :param list data: Learning data, format depends on automaton_type (see run_Alergia for details).
        :param str automaton_type: Either 'mdp', 'mc', or 'smm'.
        :param float | str eps: Epsilon value for the default HoeffdingCompatibility, or 'auto' to compute it from
            the data.
        :param CompatibilityChecker | None compatibility_checker: Custom compatibility checker, HoeffdingCompatibility
            with eps value by default.
        :param bool print_info: If True, print timing/statistics information.
        """
        assert eps == 'auto' or 0 < eps <= 2

        self.automaton_type = automaton_type
        self.print_info = print_info

        if eps == 'auto':
            eps = 10 / sum(len(d) - 1 for d in data)  # len - 1 to ignore initial output

        self.diff_checker = HoeffdingCompatibility(eps) if not compatibility_checker else compatibility_checker

        pta_start = time.time()

        self.fpta = create_fpta(data, automaton_type)

        pta_time = round(time.time() - pta_start, 2)
        if self.print_info:
            print(f'PTA Construction Time:  {pta_time}')

    def compatibility_test(self, a: AlergiaPtaNode, b: AlergiaPtaNode) -> bool:
        """
        Recursively checks whether two FPTA nodes (and all their reachable descendants) are compatible for merging.

        :param AlergiaPtaNode a: First node.
        :param AlergiaPtaNode b: Second node.
        :return bool: True if the nodes are compatible, False otherwise.
        """

        # for MDPs and MC output of the state needs to be the same
        if self.automaton_type != 'smm' and a.output != b.output:
            return False

        # leaf nodes are merged
        if not a.original_children.keys() or not b.original_children.keys():
            return True

        # if states are statistically different, do not merge
        if self.diff_checker.are_states_different(a, b):
            return False

        # check future for compatibility
        for el in set(a.original_children.keys()).intersection(b.original_children.keys()):
            if not self.compatibility_test(a.original_children[el], b.original_children[el]):
                return False

        return True

    def merge(self, red_state: AlergiaPtaNode, blue_state: AlergiaPtaNode) -> None:
        """
        Merges a blue node into a red (representative) node by rewiring its parent's transition and folding it in.

        :param AlergiaPtaNode red_state: Representative state that will absorb the blue state.
        :param AlergiaPtaNode blue_state: State to be merged into the red state.
        """
        b_prefix = blue_state.prefix
        to_update = self.fpta
        for p in b_prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[b_prefix[-1]] = red_state

        self.fold(red_state, blue_state)

    def fold(self, red: AlergiaPtaNode, blue: AlergiaPtaNode) -> None:
        """
        Recursively folds the subtree rooted at blue into the subtree rooted at red, merging input frequencies.

        :param AlergiaPtaNode red: Node into which blue is folded.
        :param AlergiaPtaNode blue: Node being folded into red.
        """
        for i, blue_child in blue.children.items():
            if i in red.children:
                red.input_frequency[i] += blue.input_frequency[i]
                self.fold(red.children[i], blue_child)
            else:
                red.children[i] = blue.children[i]
                red.input_frequency[i] = blue.input_frequency[i]

    def run(self) -> MarkovChain | Mdp | StochasticMealyMachine:
        """
        Runs the Alergia state-merging loop on the FPTA and converts the resulting red states into an automaton.

        :return MarkovChain | Mdp | StochasticMealyMachine: The learned automaton.
        """
        start_time = time.time()

        # representative nodes that will be included in the final output model
        red = [self.fpta]
        # intermediate successors scheduled for testing
        blue = self.fpta.successors()

        while blue:
            # get lexicographically minimal blue node (one with the shortest prefix)
            lex_min_blue = min(list(blue))
            merged = False

            for red_state in red:
                if self.compatibility_test(red_state, lex_min_blue):
                    self.merge(red_state, lex_min_blue)
                    merged = True
                    break

            if not merged:
                insort(red, lex_min_blue)

            blue.clear()

            for r in red:
                for s in r.successors():
                    if s not in red:
                        blue.append(s)

        assert sorted(red, key=lambda x: len(x.prefix)) == red

        self.normalize(red)

        for i, r in enumerate(red):
            r.state_id = f'q{i}'

        if self.print_info:
            print(f'Alergia Learning Time: {round(time.time() - start_time, 2)}')
            print(f'Alergia Learned {len(red)} state automaton.')

        return self.to_automaton(red)

    def normalize(self, red: list) -> None:
        """
        Normalizes input/output frequencies of all red states into probabilities.

        :param list red: List of representative (red) AlergiaPtaNode states.
        """
        red_sorted = sorted(list(red), key=lambda x: len(x.prefix))
        for r in red_sorted:
            # Initializing in here saves many unnecessary initializations
            r.children_prob = dict()
            if self.automaton_type == 'mc':
                total_output = sum(r.input_frequency.values())
                for i in r.input_frequency.keys():
                    r.children_prob[i] = r.input_frequency[i] / total_output
            else:
                for i, o in r.input_frequency.keys():
                    r.children_prob[(i, o)] = r.input_frequency[(i, o)] / r.get_input_frequency(i)

    def to_automaton(self, red: list) -> MarkovChain | Mdp | StochasticMealyMachine:
        """
        Converts the list of red FPTA nodes into an automaton of the configured type.

        :param list red: List of representative (red) AlergiaPtaNode states.
        :return MarkovChain | Mdp | StochasticMealyMachine: The constructed automaton.
        """
        s_c = state_automaton_map[self.automaton_type][0]
        a_c = state_automaton_map[self.automaton_type][1]

        states = []
        initial_state = None
        red_mdp_map = dict()
        for s in red:
            if self.automaton_type != 'smm':
                automaton_state = s_c(s.state_id, output=s.output)
            else:
                automaton_state = s_c(s.state_id)

            automaton_state.prefix = s.prefix
            states.append(automaton_state)
            red_mdp_map[tuple(s.prefix)] = automaton_state
            red_mdp_map[automaton_state.state_id] = s
            if not s.prefix:
                initial_state = automaton_state

        for s in states:
            red_eq = red_mdp_map[s.state_id]
            for io, c in red_eq.children.items():
                destination = red_mdp_map[tuple(c.prefix)]
                i = io if self.automaton_type == 'mc' else io[0]
                if self.automaton_type == 'mdp':
                    s.transitions[i].append((destination, red_eq.children_prob[io]))
                elif self.automaton_type == 'mc':
                    s.transitions.append((destination, red_eq.children_prob[i]))
                elif self.automaton_type == 'smm':
                    s.transitions[i].append((destination, io[1], red_eq.children_prob[io]))
                else:
                    s.transitions[i] = destination

        return a_c(initial_state, states)


def run_Alergia(data: list, automaton_type: str, eps: float | str = 0.05,
                compatibility_checker: CompatibilityChecker | None = None,
                print_info: bool = False) -> MarkovChain | Mdp | StochasticMealyMachine:
    """
    Run Alergia or IOAlergia on provided data.

    :param list data: Data either in a form [[I,I,I],[I,I,I],...] if learning Markov Chains or
        [[O,(I,O),(I,O)...], [O,(I,O), (I, O)_,...],..,] if learning MDPs, or [[I,O,I,O...], [I,O_,...],..,] if
        learning SMMs (I represents input, O output). Note that in whole data first symbol of each entry should be
        the same (Initial output of the MDP/MC).
    :param str automaton_type: Either 'mdp' if you wish to learn an MDP, 'mc' if you want to learn Markov Chain, or
        'smm' if you want to learn stochastic Mealy machine.
    :param float | str eps: Epsilon value if you are using default HoeffdingCompatibility. If it is set to 'auto' it
        will be computed as 10/(all steps in the data).
    :param CompatibilityChecker | None compatibility_checker: Impl. of class CompatibilityChecker,
        HoeffdingCompatibility with eps value by default (note: not interchangeable, depends on data).
    :param bool print_info: Print learning statistics.
    :return MarkovChain | Mdp | StochasticMealyMachine: Learned MDP, SMM, or Markov chain.
    """
    assert automaton_type in {'mdp', 'mc', 'smm'}
    alergia = Alergia(data, eps=eps, automaton_type=automaton_type,
                      compatibility_checker=compatibility_checker, print_info=print_info)
    model = alergia.run()
    del alergia.fpta, alergia
    return model


def run_JAlergia(path_to_data_file: str | list, automaton_type: str, path_to_jAlergia_jar: str,
                  eps: float = 0.05, heap_memory: str = '-Xmx2048M') -> MarkovChain | Mdp | StochasticMealyMachine | None:
    """
    Run Alergia or IOAlergia on provided data using the external JAlergia Java implementation.

    :param str | list path_to_data_file: Either a data in a list of lists or a path to file containing data. Form
        [[I,I,I],[I,I,I],...] if learning Markov Chains or [[O,I,O,I,O...], [O,I,O_,...],..,] if learning MDPs
        (I represents input, O output), or [[I,O,I,O...], [I,O_,...],..,] if learning SMMs. Note that in whole data
        first symbol of each entry should be the same (Initial output of the MDP/MC).
    :param str automaton_type: Either 'mdp' if you wish to learn an MDP, 'mc' if you want to learn Markov Chain, or
        'smm' if you want to learn stochastic Mealy machine.
    :param str path_to_jAlergia_jar: Path to the JAlergia jar file.
    :param float eps: Epsilon value.
    :param str heap_memory: Java heap memory flag, increase if heap is full.
    :return MarkovChain | Mdp | StochasticMealyMachine | None: Learned model, or None if an error occurred.
    """
    assert automaton_type in {'mdp', 'smm', 'mc'}

    import os
    import subprocess
    from aalpy.utils.FileHandler import load_automaton_from_file

    save_file = "jAlergiaModel.dot"
    delete_tmp_file = False
    if os.path.exists(save_file):
        os.remove(save_file)

    if os.path.exists(path_to_jAlergia_jar):
        path_to_jAlergia_jar = os.path.abspath(path_to_jAlergia_jar)
    else:
        print(f'JAlergia jar not found at {path_to_jAlergia_jar}.')
        return

    if isinstance(path_to_data_file, str):
        if os.path.exists(path_to_data_file):
            abs_path = os.path.abspath(path_to_data_file)
        else:
            print('Input file not found.')
            return
    else:
        if not isinstance(path_to_data_file, (list, tuple)):
            print('Data should be either a list of sequences or a path to the data file.')
        with open('jAlergiaInputs.txt', 'w') as f:
            for seq in path_to_data_file:
                f.write(','.join([str(s) for s in seq]) + '\n')
        delete_tmp_file = True
        abs_path = os.path.abspath('jAlergiaInputs.txt')

    subprocess.call(['java', heap_memory, '-jar', path_to_jAlergia_jar, '-input', abs_path, '-eps', str(eps), '-type',
                     automaton_type])

    if not os.path.exists(save_file):
        print("JAlergia error occurred.")
        return

    model = load_automaton_from_file(save_file, automaton_type=automaton_type)
    os.remove(save_file)
    if delete_tmp_file:
        os.remove('jAlergiaInputs.txt')

    return model
