from collections import defaultdict
from math import sqrt, log

from aalpy.automata import Mdp, MdpState, StochasticMealyState, StochasticMealyMachine
from .StochasticTeacher import StochasticTeacher, Node


def is_suffix_of(suffix, trace) -> bool:
    """

    Args:
      suffix: 
      trace: 

    Returns:

    """
    if len(trace) < len(suffix):
        return False
    else:
        return trace[-len(suffix):] == suffix


class SamplingBasedObservationTable:
    def __init__(self, input_alphabet: list, automaton_type, teacher: StochasticTeacher, alpha=0.05, strategy='normal'):
        """Constructor of the observation table. Initial queries are asked in the constructor.

        Args:

          input_alphabet: input alphabet
          teacher: stochastic teacher
          alpha: constant used in Hoeffding bound

        """
        assert input_alphabet is not None and teacher is not None
        self.automaton_type = automaton_type

        self.input_alphabet = [tuple([a]) for a in input_alphabet]

        self.S = list()  # prefixes of S
        self.E = [tuple([a]) for a in input_alphabet]
        self.T = defaultdict(dict)

        self.teacher = teacher
        self.empty_word = tuple()
        self.alpha = alpha
        self.strategy = strategy

        # initial output
        if automaton_type == 'mdp':
            self.initial_output = tuple(teacher.initial_value)
            self.S.append(self.initial_output)
        else:
            self.S.append(tuple())

        # Cache
        self.compatibility_classes_representatives = None
        self.compatibility_class_rows = dict()
        self.freq_query_cache = dict()

    def refine_not_completed_cells(self, n_resample, uniform=False):
        """Firstly a prefix-tree acceptor is constructed for all non-completed cells and then that tree is used
        for online testing/sampling.

        Args:
          uniform: if true, all cells will be uniformly sampled (Default value = False)
          n_resample: Number of resamples

        Returns:

        """
        if self.automaton_type == 'mdp':
            pta_root = Node(self.initial_output[0])
        else:
            pta_root = Node(None)

        if self.strategy == 'normal':
            to_refine = []
            for s in self.S + list(self.get_extended_s()):
                for e in self.E:
                    if not self.teacher.complete_query(s, e):
                        to_refine.append(s + e)

            if not to_refine:
                return

            to_refine.sort(key=len, reverse=True)

            for trace in to_refine:
                self.add_to_PTA(pta_root, trace)

        else:
            for s in self.S + list(self.get_extended_s()):
                if uniform:
                    for e in self.E:
                        self.add_to_PTA(pta_root, s + e, 1)
                else:
                    for e in self.E:
                        longest_row_trace_prefix = (s + e)[:-1]
                        while longest_row_trace_prefix not in self.T.keys():
                            longest_row_trace_prefix = longest_row_trace_prefix[:-1]
                        row_repr = 0
                        for r in self.compatibility_classes_representatives:
                            if self.are_rows_compatible(longest_row_trace_prefix, r):
                                row_repr += 1
                        # row_repr can be zero for non-closed
                        uncertainty_value = max((row_repr - 1) * 2, 1)
                        self.add_to_PTA(pta_root, s + e, uncertainty_value)

        for i in range(n_resample):
            self.teacher.refine_query(pta_root)

    def update_obs_table_with_freq_obs(self, element_of_s=None):
        """Updates cells in the observation table with frequency data. If the row in S has no extension yet, it is
        generated and its cells populated.

        Args:
          element_of_s: if not None, selected row and its extensions will be updated (Default value = None)

        Returns:

        """
        if element_of_s:
            s_set = element_of_s + list(self.get_extended_s(element_of_s=element_of_s))
        else:
            s_set = self.S + list(self.get_extended_s())
        # s_set = element_of_s if  else self.S + list(self.get_extended_s())

        for s in s_set:
            for e in self.E:
                self.T[s][e] = self.teacher.frequency_query(s, e)
                self.freq_query_cache[s + e] = self.T[s][e]

    def get_extended_s(self, element_of_s=None):
        """Generator returning all elements of the extended S set.

        Args:
          element_of_s:  (Default value = None)

        Returns:

        """
        s_set = element_of_s if element_of_s else self.S
        for s in s_set:
            for i in self.input_alphabet:
                if s + i in self.freq_query_cache.keys():
                    freq_dict = self.freq_query_cache[s + i]
                else:
                    freq_dict = self.teacher.frequency_query(s, i)
                for out, freq in freq_dict.items():
                    new_pref = s + i + tuple([out])
                    if freq > 0:
                        yield new_pref

    def make_closed_and_consistent(self):
        """Observation table is updated until it is closed and consistent. Note that due the updated notion of row
        equivalence no sampling is needed.

        Args:

        Returns:

        """
        self.update_compatibility_classes()

        while True:
            closed, consistent = False, False
            row_to_close = self.get_row_to_close()
            if not row_to_close:
                closed = True
            if row_to_close:
                self.S.append(row_to_close)
                self.update_obs_table_with_freq_obs(element_of_s=[row_to_close])
                self.update_compatibility_classes()

            consistency_violation = self.get_consistency_violation()
            if not consistency_violation:
                consistent = True
            if consistency_violation:
                if consistency_violation not in self.E:
                    self.E.append(consistency_violation)
                self.update_obs_table_with_freq_obs()
                self.update_compatibility_classes()

            if closed and consistent:
                break

    def get_row_to_close(self):
        """Returns a row that is not closed.
        :return: single row that is not closed

        Args:

        Returns:

        """
        for lt in self.get_extended_s():
            row_is_closed = False
            for r in self.compatibility_classes_representatives:
                if self.are_rows_compatible(r, lt):
                    row_is_closed = True
                    break
            if not row_is_closed:
                return lt
        return None

    def get_consistency_violation(self, ignore=None):
        """Find and return cause of consistency violation. Only computed on the compatibility class representatives.
        :return: element of input + element of output + element of e that lead to the inconsistency

        Args:
          ignore:  (Default value = None)

        Returns:

        """
        for ind, s1 in enumerate(self.compatibility_classes_representatives):
            for s2 in self.compatibility_class_rows[s1]:
                #if self.are_rows_compatible(s1, s2, ignore):
                i_o_pairs = [(i, tuple([o])) for i in self.input_alphabet for o in self.T[s1][i].keys()]
                for i, o in i_o_pairs:
                    s1_keys = self.T[s1 + i + o].keys()
                    s2_keys = self.T[s2 + i + o].keys()
                    if not s1_keys or not s2_keys:
                        continue

                    for e in self.E:
                        if e == ignore:
                            continue
                        if self.cell_diff(s1 + i + o, s2 + i + o, e):
                            return i + o + e
        return None

    def get_representative(self, target):
        """

        Args:
          target: row in the observation table

        Returns:
          a representative compatible with the target

        """
        for r in self.compatibility_classes_representatives:
            if self.are_rows_compatible(r, target):
                return r
        return None

    def trim_columns(self):
        """ """
        reverse_sorted_E = list(self.E)
        reverse_sorted_E.sort(key=len, reverse=True)
        to_remove = []
        to_keep = []
        self.update_obs_table_with_freq_obs()
        self.make_closed_and_consistent()  # need a closed observation table

        for e in reverse_sorted_E:
            if e in self.input_alphabet:
                continue
            contains_dependent = False
            for other_e in to_keep:
                if is_suffix_of(e, other_e):
                    contains_dependent = True
            if contains_dependent:
                to_keep.append(e)
            elif self.get_consistency_violation(e):
                to_keep.append(e)
            else:
                self.E.remove(e)  # need to remove here for get_consistency_violation to work
                to_remove.append(e)

        for e in to_remove:
            for s in self.T.keys():
                if e in self.T[s]:
                    self.T[s].pop(e)

    def trim(self, hypothesis):
        """Removes unnecessary rows from the observation table.

        Args:
          hypothesis: 

        Returns:

        """

        prefix_to_state_dict = {state.prefix: state for state in hypothesis.states}

        to_remove = []
        for s in self.S:
            if s in self.compatibility_classes_representatives or s in to_remove:
                continue

            rep = self.get_representative(s)
            if self.automaton_type == 'mdp':
                if 'chaos' in {t[0].output for transitions in prefix_to_state_dict[rep].transitions.values()
                               for t in transitions}:
                    continue
            else:
                if 'chaos' in {t[1] for transitions in prefix_to_state_dict[rep].transitions.values()
                               for t in transitions}:
                    continue

            num_compatible_repr = 0
            row_is_prefix = False
            for r in self.compatibility_classes_representatives:
                if self.are_rows_compatible(r, s):
                    num_compatible_repr += 1
                if len(s) < len(r) and s == r[:len(s)]:
                    row_is_prefix = True
                    continue
            if num_compatible_repr != 1 or row_is_prefix:
                continue

            to_remove.append(s)
            for otherS in self.S:
                if s == otherS[:len(s)] and otherS not in to_remove:
                    to_remove.append(otherS)

        for s in to_remove:
            self.S.remove(s)
            self.T.pop(s, None)
            for i in self.input_alphabet:
                for o in self.T[s + i]:
                    self.T.pop(s + i + o, None)

        self.trim_columns()

    def stop(self, learning_round, chaos_present, min_rounds=5, max_rounds=None,
             target_unambiguity=0.99, print_unambiguity=False):
        """Decide if learning should terminate.

        Args:
          learning_round: current learning round
          chaos_present: True if chaos counterexample existed in current learning round
          min_rounds: minimum number of learning rounds (Default value = 5)
          max_rounds: maximum number of learning rounds (Default value = None)
          target_unambiguity: percentage of rows with unambiguous representatives (Default value = 0.99)
          print_unambiguity: if true, current unambiguity rate will be printed (Default value = False)

        Returns:
          True if stopping condition satisfied, false otherwise

        """
        if max_rounds:
            assert min_rounds <= max_rounds
        if max_rounds and learning_round == max_rounds:
            return True

        if chaos_present:
            return False

        extended_s = list(self.get_extended_s())
        self.update_compatibility_classes()
        numerator = 0
        for row in self.S + extended_s:
            row_repr = 0
            for r in self.compatibility_classes_representatives:
                if self.are_rows_compatible(row, r):
                    row_repr += 1
            numerator += 1 if row_repr == 1 else 0

        unambiguous_rows_percentage = numerator / len(self.S + extended_s)
        if print_unambiguity and learning_round % 5 == 0:
            print(f'Unambiguous rows: {round(unambiguous_rows_percentage * 100, 2)}%;'
                  f' {numerator} out of {len(self.S + extended_s)}')
        if learning_round >= min_rounds and unambiguous_rows_percentage >= target_unambiguity:
            return True

    def cell_diff(self, s1, s2, e):
        """

        Args:
          s1: prefix of row s1
          s2: prefix of row s2
          e: element of E

        Returns:
          True if cells are different, false otherwise

        """
        if self.strategy == 'normal':
            if self.teacher.complete_query(s1, e) and self.teacher.complete_query(s2, e):
                c1 = self.T[s1][e]
                c2 = self.T[s2][e]

                if set(c1.keys()) != set(c2.keys()):
                    return True

                n1 = sum(c1.values())
                n2 = sum(c2.values())

                if n1 > 0 and n2 > 0:
                    for o in c1.keys():
                        if abs(c1[o] / n1 - c2[o] / n2) > \
                                ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / self.alpha))):
                            return True
        else:
            if e in self.T[s1] and e in self.T[s2]:
                c1 = self.T[s1][e]
                c2 = self.T[s2][e]

                n1 = sum(c1.values())
                n2 = sum(c2.values())

                if n1 > 0 and n2 > 0:
                    for o in set(c1.keys()).union(c2.keys()):
                        c1o = c1[o] if o in c1.keys() else 0
                        c2o = c2[o] if o in c2.keys() else 0
                        alpha1 = 0.05 
                        alpha2 = 0.05 
                        epsilon1 = sqrt((1. / (2 * n1)) * log(2. / alpha1))
                        epsilon2 = sqrt((1. / (2 * n2)) * log(2. / alpha2))

                        if abs(c1o / n1 - c2o / n2) > epsilon1 + epsilon2:
                            return True
        return False

    def are_rows_compatible(self, s1, s2, e_ignore=None):
        """Check if the rows are compatible.
        Rows are compatible if all cells are compatible(not different) and their prefixes
        end in the same output element.

        Args:
          s1: prefix of row s1
          s2: prefix of row s2
          e_ignore: e not considered for the computation of row compatibility (Default value = None)

        Returns:
          True if rows are compatible, False otherwise

        """
        if self.automaton_type == 'mdp' and s1[-1] != s2[-1]:
            return False

        for e in self.E:
            if e == e_ignore:
                continue
            if self.cell_diff(s1, s2, e):
                return False
        return True

    def update_compatibility_classes(self):
        """Updates the compatibility classes and stores their representatives."""
        self.compatibility_class_rows.clear()
        class_rank_pair = []
        for s in self.S:
            rank = sum([sum(self.T[s][i].values()) for i in self.input_alphabet])
            class_rank_pair.append((s, rank))

        class_rank_pair.sort(key=lambda x: x[1], reverse=True)
        compatibility_classes = [c[0] for c in class_rank_pair]

        tmp_classes = list(compatibility_classes)
        not_partitioned = list(self.S)

        representatives = []
        while not_partitioned:
            r = tmp_classes.pop(0)
            not_partitioned.remove(r)

            cg_r = [s for s in not_partitioned if self.are_rows_compatible(r, s)]

            comp_rows = []
            for s in self.S:
                if s not in cg_r and self.are_rows_compatible(r, s):
                    comp_rows.append(s)

            self.compatibility_class_rows[r] = cg_r + comp_rows

            representatives.append(r)
            for sp in cg_r:
                not_partitioned.remove(sp)
                tmp_classes.remove(sp)

        self.compatibility_classes_representatives = representatives

    def chaos_counterexample(self, hypothesis: Mdp):
        """If chaos state is reachable, return path to chaos

        Args:
          hypothesis: current hypothesis
          hypothesis: Mdp: 

        Returns:
          prefix leading to a state from which chaos state is reachable

        """
        for state in hypothesis.states:
            for i in self.input_alphabet:
                output_states = state.transitions[i[0]]
                if self.automaton_type == 'mdp':
                    for (s, _) in output_states:
                        if s.output == 'chaos':
                            return state.prefix
                else:
                    for (_, o, _) in output_states:
                        if o == 'chaos':
                            return state.prefix
        return None

    def add_to_PTA(self, pta_root, trace, uncertainty_value=None):
        """Adds a trace to the PTA. PTA is later used for online sampling. The uncertainty value is added to inputs as
        frequencies, which specify how often a particular input should be sampled.

        Args:
          pta_root: root of the prefix tree acceptor
          trace: trace to add to the PTA
          uncertainty_value: uncertainty value (Default value = None)

        Returns:

        """
        curr_node = pta_root
        start = 1 if self.automaton_type == 'mdp' else 0
        for index in range(start, len(trace), 2):
            inp = trace[index]
            if uncertainty_value:
                # use frequencies for uncertainties
                curr_node.input_frequencies[inp] += uncertainty_value
            # need to add a dummy output in the leaves
            output = trace[index + 1] if index + 1 < len(trace) else "dummy"
            child = curr_node.get_child(inp, output)
            if child:
                curr_node = child
            else:
                new_node = Node(output)
                curr_node.children[inp].append(new_node)
                curr_node = new_node

    def generate_hypothesis(self):
        """Generates the hypothesis from the observation table.
        :return: current hypothesis

        Args:

        Returns:

        """
        r_state_map = dict()
        state_counter = 0
        for r in self.compatibility_classes_representatives:
            if self.automaton_type == 'mdp':
                r_state_map[r] = MdpState(state_id=f's{state_counter}', output=r[-1])
            else:
                r_state_map[r] = StochasticMealyState(state_id=f's{state_counter}')
            r_state_map[r].prefix = r
            state_counter += 1
        if self.automaton_type == 'mdp':
            r_state_map['chaos'] = MdpState(state_id=f's{state_counter}', output='chaos')
            for i in self.input_alphabet:
                r_state_map['chaos'].transitions[i[0]].append((r_state_map['chaos'], 1.))
        else:
            r_state_map['chaos'] = StochasticMealyState(state_id=f'chaos')
            for i in self.input_alphabet:
                r_state_map['chaos'].transitions[i[0]].append((r_state_map['chaos'], 'chaos', 1.))

        for s in self.compatibility_classes_representatives:
            for i in self.input_alphabet:
                freq_dict = self.T[s][i]

                total_sum = sum(freq_dict.values())

                origin_state = s
                if self.strategy == 'normal' and not self.teacher.complete_query(s, i) or self.strategy != 'normal' and\
                        i not in self.T[s]:  # if not self.teacher.complete_query(s, i):
                    if self.automaton_type == 'mdp':
                        r_state_map[origin_state].transitions[i[0]].append((r_state_map['chaos'], 1.))
                    else:
                        r_state_map[origin_state].transitions[i[0]].append((r_state_map['chaos'], 'chaos', 1.))
                else:
                    if len(freq_dict.items()) == 0:
                        if self.automaton_type == 'mdp':
                            r_state_map[origin_state].transitions[i[0]].append((r_state_map['chaos'], 1.))
                        else:
                            r_state_map[origin_state].transitions[i[0]].append((r_state_map['chaos'], 'chaos', 1.))
                    else:
                        for output, frequency in freq_dict.items():
                            new_state = self.get_representative(s + i + tuple([output]))
                            if self.automaton_type == 'mdp':
                                r_state_map[origin_state].transitions[i[0]].append(
                                    (r_state_map[new_state], frequency / total_sum))
                            else:
                                r_state_map[origin_state].transitions[i[0]].append(
                                    (r_state_map[new_state], output, frequency / total_sum))

        if self.automaton_type == 'mdp':
            return Mdp(r_state_map[self.get_representative(self.initial_output)], list(r_state_map.values()))
        else:
            return StochasticMealyMachine(r_state_map[tuple()], list(r_state_map.values()))
