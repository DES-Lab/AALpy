from collections import defaultdict

from aalpy.automata import MarkovChain, MdpState, Mdp, McState
from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import HoeffdingCompatibility
from aalpy.learning_algs.stochastic_passive.DataHandler import create_fpta


class Alergia:
    def __init__(self, data, is_iofpta=False, eps=0.05, compatibility_checker=None):
        assert 0 < eps <= 2
        self.t, self.a = create_fpta(data, is_iofpta)
        self.is_iofpta = is_iofpta
        self.diff_checker = HoeffdingCompatibility(eps) if not compatibility_checker else compatibility_checker

    def compatibility_test(self, a, b):
        if a.output != b.output:
            return False

        if not a.children.values() or not b.children.values():
            return True

        if not self.diff_checker.check_difference(a, b):
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
                q_r.children[i].frequency += c.frequency
                self.fold(q_r.children[i], c)
            else:
                q_r.children[i] = c

    def run(self):

        red = {self.a}  # representative nodes and will be included in the final output model
        blue = self.a.succs()  # intermediate successors scheduled for testing

        while blue:

            lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))

            red_sorted = sorted(list(red), key=lambda x: len(x.prefix))

            merged = False

            for q_r in red_sorted:
                if self.compatibility_test(self.get_blue_node(q_r), self.get_blue_node(lex_min_blue)):
                    self.merge(q_r, lex_min_blue)
                    merged = True
                    break

            if not merged:
                red.add(lex_min_blue)

            blue.clear()
            prefixes_in_red = [s.prefix for s in red]
            for r in red:
                for s in r.succs():
                    if s.prefix not in prefixes_in_red:
                        blue.append(s)

        red = sorted(list(red), key=lambda x: len(x.prefix))

        self.normalize(red)

        for i, r in enumerate(red):
            r.state_id = f'q{i}'

        return self.to_automaton(red)

    def normalize(self, red):
        red_sorted = sorted(list(red), key=lambda x: len(x.prefix))
        for r in red_sorted:
            if not self.is_iofpta:
                total_output = sum([c.frequency for c in r.children.values()])
                for i, c in r.children.items():
                    r.children_prob[i] = c.frequency / total_output
            else:
                outputs_per_input = defaultdict(int)
                for io, c in r.children.items():
                    outputs_per_input[io[0]] += c.frequency
                for io, c in r.children.items():
                    r.children_prob[io] = c.frequency / outputs_per_input[io[0]]

    def get_blue_node(self, red_node):
        blue = self.t
        for p in red_node.prefix:
            blue = blue.children[p]
        return blue

    def to_automaton(self, red):
        s_c = MdpState if self.is_iofpta else McState
        a_c = Mdp if self.is_iofpta else MarkovChain

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
                i = io[0] if self.is_iofpta else io
                s.transitions[i].append((destination, red_eq.children_prob[io]))

        return a_c(initial_state, states)


def run_Alergia(data, eps=0.05, is_iofpta=False):
    alergia = Alergia(data, eps=eps, is_iofpta=is_iofpta)
    model = alergia.run()
    return model
