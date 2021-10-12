import time
from collections import defaultdict
from bisect import insort

from aalpy.automata import MarkovChain, MdpState, Mdp, McState, MooreMachine, MooreState
from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import HoeffdingCompatibility
from aalpy.learning_algs.stochastic_passive.FPTA import create_fpta

state_automaton_map = {'mc': (McState, MarkovChain), 'mdp': (MdpState, Mdp), 'moore': (MooreState, MooreMachine)}


class Alergia:
    def __init__(self, data, automaton_type, eps=0.005, compatibility_checker=None, print_info=False):
        assert eps == 'auto' or 0 < eps <= 2

        self.automaton_type = automaton_type
        self.print_info = print_info

        if eps == 'auto':
            eps = 10 / sum(len(d) - 1 for d in data)  # len - 1 to ignore initial output

        self.diff_checker = HoeffdingCompatibility(eps) if not compatibility_checker else compatibility_checker

        pta_start = time.time()

        self.t, self.a = create_fpta(data, automaton_type)

        pta_time = round(time.time() - pta_start, 2)
        if self.print_info:
            print(f'PTA Construction Time:  {pta_time}')

    def compatibility_test(self, a, b):
        if a.output != b.output:
            return False

        if not a.children.values() or not b.children.values():
            return True

        if self.automaton_type != 'moore' and not self.diff_checker.check_difference(a, b):
            return False

        for el in set(a.children.keys()).intersection(b.children.keys()):
            if not self.compatibility_test(a.children[el], b.children[el]):
                return False

        return True

    def merge(self, q_r, q_b):
        t_q_b = self.get_blue_node(q_b)
        prefix_leading_to_state = q_b.prefix[:-1]
        to_update = self.a
        for p in prefix_leading_to_state:
            to_update = to_update.children[p]

        to_update.children[q_b.prefix[-1]] = q_r

        self.fold(q_r, t_q_b)

    def fold(self, q_r, q_b):
        for i, c in q_b.children.items():
            if i in q_r.children.keys():
                q_r.input_frequency[i] += c.input_frequency[i]
                self.fold(q_r.children[i], c)
            else:
                q_r.children[i] = c.copy()
                q_r.input_frequency[i] = q_b.input_frequency[i]

    def run(self):
        start_time = time.time()

        red = [self.a]  # representative nodes and will be included in the final output model
        blue = self.a.succs()  # intermediate successors scheduled for testing

        while blue:
            lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))
            merged = False

            for q_r in red:
                if self.compatibility_test(self.get_blue_node(q_r), self.get_blue_node(lex_min_blue)):
                    self.merge(q_r, lex_min_blue)
                    merged = True
                    break

            if not merged:
                insort(red, lex_min_blue)

            blue.clear()
            prefixes_in_red = [s.prefix for s in red]
            for r in red:
                for s in r.succs():
                    if s.prefix not in prefixes_in_red:
                        blue.append(s)

        assert sorted(red, key=lambda x: len(x.prefix)) == red

        if self.automaton_type != 'moore':
            self.normalize(red)

        for i, r in enumerate(red):
            r.state_id = f'q{i}'

        if self.print_info:
            print(f'Alergia Learning Time: {round(time.time() - start_time, 2)}')
            print(f'Alergia Learned {len(red)} state automaton.')
        return self.to_automaton(red)

    def normalize(self, red):
        red_sorted = sorted(list(red), key=lambda x: len(x.prefix))
        for r in red_sorted:
            if self.automaton_type == 'mc':
                total_output = sum(r.input_frequency.values())
                for i in r.input_frequency.keys():
                    r.children_prob[i] = r.input_frequency[i] / total_output
            else:
                outputs_per_input = defaultdict(int)
                for io, freq in r.input_frequency.items():
                    outputs_per_input[io[0]] += freq
                for io in r.input_frequency.keys():
                    r.children_prob[io] = r.input_frequency[io] / outputs_per_input[io[0]]

    def get_blue_node(self, red_node):
        blue = self.t
        for p in red_node.prefix:
            blue = blue.children[p]
        return blue

    def to_automaton(self, red):
        s_c = state_automaton_map[self.automaton_type][0]
        a_c = state_automaton_map[self.automaton_type][1]

        states = []
        initial_state = None
        red_mdp_map = dict()
        for s in red:
            automaton_state = s_c(s.state_id, output=s.output)
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
                else:
                    s.transitions[i] = destination

        return a_c(initial_state, states)


def run_Alergia(data, automaton_type, eps=0.005, compatibility_checker=None, print_info=False):
    """
    Run Alergia or IOAlergia on provided data.

    Args:

        data: data either in a form [[I,I,I],[I,I,I],...] if learning Markov Chains or [[O,(I,O),(I,O)...],
        [O,(I,O_,...],..,] if learning MDPs (I represents input, O output).
        Note that in whole data first symbol of each entry should be the same (Initial output of the MDP/MC).

        eps: epsilon value if you are using default HoeffdingCompatibility. If it is set to 'auto' it will be computed
        as 10/(all steps in the data)

        automaton_type: either 'mdp' if you wish to learn an MDP, 'mc' if you want to learn Markov Chain, and 'moore'
                        if you want to learn Moore Machine (underlying structure is deterministic)

        compatibility_checker: impl. of class CompatibilityChecker, HoeffdingCompatibility with eps value by default

        (note: not interchangeable, depends on data)
        print_info:

    Returns:

        mdp or markov chain
    """
    assert automaton_type in {'mdp', 'mc', 'moore'}
    alergia = Alergia(data, eps=eps, automaton_type=automaton_type,
                      compatibility_checker=compatibility_checker, print_info=print_info)
    model = alergia.run()
    return model
