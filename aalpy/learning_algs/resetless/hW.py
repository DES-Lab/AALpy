import time
from collections import defaultdict, deque
from random import choice

from aalpy.automata import MealyState, MealyMachine
from aalpy.utils.HelperFunctions import all_suffixes, print_learning_info


class ModelState:
    def __init__(self, hs):
        self.hs = hs
        self.state_w_values = {}

        self.transitions = {}
        self.output_fun = {}

        self.transition_w_values = {}
        self.learned_w_per_input = defaultdict(set)


class hW:
    def __init__(self, input_al, sul,
                 num_testing_steps=200,
                 reset_testing_counter=True,
                 query_for_initial_state=False):

        self.input_alphabet = input_al # to ensure determinism if passed as set
        self.sul = sul
        self.total_testing_steps = 0
        self.num_testing_steps = num_testing_steps
        self.reset_testing_counter = reset_testing_counter
        self.query_for_initial_state = query_for_initial_state

        self.homing_sequence = ()
        self.W = []

        self.global_trace = []

        self.interrupt = False

        self.state_map = {}                      # full W-profile -> ModelState
        self.h_response_map = {}                 # h-response -> partial or complete ModelState
        self.dictionary = {}                     # (h_response, alpha, x, w) -> (alpha_out, x_out, w_out)
        self.h_w_dictionary = {}                 # (h, h_response, w) -> w_out
        self._conjecture_probe_seen = set()

        # Incremental h-ND index
        self._hs_cont_starts = defaultdict(list)   # hs_output -> [cont_start_pos, ...] (ordered for determinism)
        self._pair_progress = {}                   # (p1, p2) -> steps compared
        self._h_nd_scan_pos = 0                    # how far global_trace has been scanned

        # reset/initialize the SUL
        sul.pre()

    def add_to_W(self, sequence, preserve_h=True):
        sequence = tuple(sequence)
        if sequence in self.W:
            return False

        self.W.append(sequence)
        self.W = [
            w for w in self.W
            if w == sequence
            or (preserve_h and w == self.homing_sequence)
            or sequence[:len(w)] != w
        ]
        return True

    def add_h_to_W(self):
        h = self.homing_sequence
        if h in self.W:
            return False
        if any(w[:len(h)] == h for w in self.W):
            return False
        return self.add_to_W(h, preserve_h=False)

    def execute_homing_sequence(self):
        observed_output = [self.step_wrapper(i) for i in self.homing_sequence]
        self.check_h_ND_consistency()
        return tuple(observed_output)

    def execute_sequence(self, seq_under_test: tuple):
        observed_output = []
        for i in seq_under_test:
            observed_output.append(self.step_wrapper(i))
        return tuple(observed_output)

    def execute_conjecture_path(self, start_state, path):
        state = start_state
        observed_outputs = []
        for pos, i in enumerate(path):
            observed_output = self.step_wrapper(i)
            observed_outputs.append(observed_output)
            expected_output = state.output_fun.get(i)
            if expected_output != observed_output:
                suffix = path[:pos + 1]
                if self.add_to_W(suffix):
                    self._reset_state_data()
                    self.interrupt = True
                else:
                    state.transitions.pop(i, None)
                    state.output_fun.pop(i, None)
                    state.learned_w_per_input[i].clear()
                    for key in list(state.transition_w_values):
                        if key[0] == i:
                            del state.transition_w_values[key]
                return None
            state = state.transitions[i]
        return state, tuple(observed_outputs)

    def create_daisy_hypothesis(self):
        state = MealyState('s0')
        for i in self.input_alphabet:
            o = self.step_wrapper(i)
            state.transitions[i] = state
            state.output_fun[i] = o
        mm = MealyMachine(state, [state])
        mm.current_state = mm.states[0]
        return mm

    def _reset_h_nd_index(self):
        self._hs_cont_starts.clear()
        self._pair_progress.clear()
        self._h_nd_scan_pos = len(self.global_trace)

    def _reset_state_data(self):
        self.state_map.clear()
        self.h_response_map.clear()
        self._conjecture_probe_seen.clear()
        self._reset_h_nd_index()

    def _all_states(self):
        seen = set()
        states = []
        for state in list(self.state_map.values()) + list(self.h_response_map.values()):
            if id(state) in seen:
                continue
            seen.add(id(state))
            states.append(state)
        return states

    def check_h_ND_consistency(self):
        h_len = len(self.homing_sequence)
        if h_len == 0:
            return True

        trace = self.global_trace
        trace_len = len(trace)

        # Scan newly added trace for hs occurrences. When a new continuation is
        # found, eagerly register pairs (new, existing) in _pair_progress so the
        # pair-comparison loop below never has to rebuild them from cont_starts.
        scan_start = max(0, self._h_nd_scan_pos - h_len + 1)
        h = self.homing_sequence
        for i in range(scan_start, trace_len - h_len + 1):
            for j in range(h_len):
                if trace[i + j][0] != h[j]:
                    break
            else:
                prefix_outputs = tuple(trace[i + j][1] for j in range(h_len))
                new_cont = i + h_len
                if new_cont not in self._hs_cont_starts[prefix_outputs]:
                    for existing in self._hs_cont_starts[prefix_outputs]:
                        p1, p2 = (new_cont, existing) if new_cont < existing else (existing, new_cont)
                        self._pair_progress[(p1, p2)] = 0
                    self._hs_cont_starts[prefix_outputs].append(new_cont)
        self._h_nd_scan_pos = trace_len

        # Advance every active pair. Diverged pairs are deleted rather than
        # marked -1, so this loop only touches pairs that still need work.
        to_delete = []
        for (p1, p2), already in self._pair_progress.items():
            compare_len = min(trace_len - p1, trace_len - p2)
            for k in range(already, compare_len):
                inp1, out1 = trace[p1 + k]
                inp2, out2 = trace[p2 + k]
                if inp1 != inp2:
                    to_delete.append((p1, p2))
                    break
                if out1 != out2:
                    self.homing_sequence += tuple(inp for inp, _ in trace[p1:p1 + k + 1])
                    self.interrupt = True
                    self._reset_state_data()
                    self.add_h_to_W()
                    return False
            else:
                if compare_len > already:
                    self._pair_progress[(p1, p2)] = compare_len

        for pair in to_delete:
            del self._pair_progress[pair]

        return True

    def check_w_ND_from_state_map(self):
        states = [(state.hs, state) for state in self._all_states()]
        for i in range(len(states)):
            hs1, s1 = states[i]
            if len(s1.state_w_values) != len(self.W):
                continue
            for j in range(i + 1, len(states)):
                hs2, s2 = states[j]
                if len(s2.state_w_values) != len(self.W):
                    continue
                if s1.state_w_values == s2.state_w_values and hs1 != hs2:
                    for k in range(min(len(hs1), len(hs2))):
                        if hs1[k] != hs2[k]:
                            new_w = self.homing_sequence[k:]
                            if new_w and new_w not in self.W:
                                self.add_to_W(new_w)
                                self._reset_state_data()
                                self.interrupt = True
                                return False
                            break
        return True

    def step_wrapper(self, letter):
        output = self.sul.step(letter)
        if isinstance(output, tuple) and len(output) == 1:
            output = output[0]
        self.global_trace.append((letter, output))
        self.sul.num_steps += 1
        return output

    def find_counterexample(self, hypothesis):
        cex = []

        if self.reset_testing_counter:
            current_test_steps = self.num_testing_steps
        else:
            current_test_steps = max(self.num_testing_steps - self.total_testing_steps, 0)

        for _ in range(current_test_steps):
            self.total_testing_steps += 1
            random_input = choice(self.input_alphabet)
            cex.append(random_input)
            o_sul = self.step_wrapper(random_input)
            o_hyp = hypothesis.step(random_input)

            if o_sul != o_hyp:
                return cex

            if self.interrupt:
                return None

        return None

    def is_complete(self):
        if not self.state_map:
            return False

        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                if i not in state.transitions:
                    return False
                if not isinstance(state.transitions[i], ModelState):
                    return False
                if state.transitions[i].hs not in self.state_map:
                    return False
        return True

    def _first_missing_transition_query(self, state):
        for i in self.input_alphabet:
            learned = state.learned_w_per_input[i]
            if len(learned) == len(self.W):
                continue
            for w in self.W:
                if w not in learned:
                    return i, w
        return None

    def find_reachable_incomplete_transition(self, start_state):
        queue = deque([(start_state, ())])
        visited = set()

        while queue:
            state, path = queue.popleft()
            if state in visited:
                continue

            visited.add(state)

            missing = self._first_missing_transition_query(state)
            if missing is not None:
                x, w = missing
                return path, state, x, w

            for input_sequence, next_state in state.transitions.items():
                queue.append((next_state, path + (input_sequence,)))

        return None

    def _reachable_state_paths(self, start_state):
        queue = deque([(start_state, ())])
        visited = set()
        reachable = []

        while queue:
            state, path = queue.popleft()
            if state in visited:
                continue
            visited.add(state)
            reachable.append((state, path))

            for i, next_state in state.transitions.items():
                if isinstance(next_state, ModelState):
                    queue.append((next_state, path + (i,)))

        return reachable

    def _simulate_from_state(self, state, sequence):
        outputs = []
        for i in sequence:
            if i not in state.output_fun or i not in state.transitions:
                return None, None
            outputs.append(state.output_fun[i])
            state = state.transitions[i]
        return tuple(outputs), state

    def _apply_conjecture_probe(self, current_state, path, w, kind):
        key = (kind, current_state.hs, path, w, self.homing_sequence)
        if key in self._conjecture_probe_seen:
            return False
        self._conjecture_probe_seen.add(key)

        alpha_result = self.execute_conjecture_path(current_state, path)
        if self.interrupt:
            return True
        if alpha_result is None:
            return True

        self.execute_sequence(self.homing_sequence)
        self.check_h_ND_consistency()
        if self.interrupt:
            return True

        self.execute_sequence(w)
        self.check_h_ND_consistency()
        return True

    def check_conjecture_inconsistencies(self, current_state):
        reachable = self._reachable_state_paths(current_state)

        for state, path in reachable:
            h_response, state_after_h = self._simulate_from_state(state, self.homing_sequence)
            if h_response is None:
                continue

            mapped_state = self.h_response_map.get(h_response)
            if mapped_state is None or len(mapped_state.state_w_values) != len(self.W):
                continue

            for w in self.W:
                expected = mapped_state.state_w_values.get(w)
                actual, _ = self._simulate_from_state(state_after_h, w)
                if actual is not None and expected != actual:
                    return self._apply_conjecture_probe(current_state, path, w, 'H')

        states_after_h = []
        for state, path in reachable:
            h_response, state_after_h = self._simulate_from_state(state, self.homing_sequence)
            if h_response is not None and state_after_h is not None:
                states_after_h.append((h_response, state_after_h, path))

        for i in range(len(states_after_h)):
            h_response, state_after_h, path = states_after_h[i]
            for j in range(i + 1, len(states_after_h)):
                other_h_response, other_after_h, other_path = states_after_h[j]
                if h_response != other_h_response or state_after_h is other_after_h:
                    continue

                for w in self.W:
                    out1, _ = self._simulate_from_state(state_after_h, w)
                    out2, _ = self._simulate_from_state(other_after_h, w)
                    if out1 is None or out2 is None or out1 == out2:
                        continue

                    if self._apply_conjecture_probe(current_state, path, w, 'h1'):
                        return True
                    if self._apply_conjecture_probe(current_state, other_path, w, 'h2'):
                        return True

        return False

    def _w_profile(self, w_values):
        return tuple(sorted(w_values.items()))

    def _find_state_by_w_values(self, w_values, new_states=None):
        profile = self._w_profile(w_values)
        matched = self.state_map.get(profile)
        if matched is not None:
            return matched

        if new_states:
            matched = new_states.get(profile)
            if matched is not None:
                return matched

        for destination_state in self.h_response_map.values():
            if w_values == destination_state.state_w_values:
                return destination_state

        return None

    def _state_for_w_values(self, w_values, new_states=None):
        matched = self._find_state_by_w_values(w_values, new_states)
        if matched is not None:
            return matched

        new_key = self._w_profile(w_values)
        matched = self.state_map.get(new_key)
        if matched is None and new_states:
            matched = new_states.get(new_key)
        if matched is not None:
            return matched

        state = ModelState(new_key)
        state.state_w_values = dict(w_values)
        if new_states is not None:
            new_states[new_key] = state
        else:
            self.state_map[new_key] = state
        return state

    def _merge_states(self, canonical, duplicate):
        for (i, w), out in duplicate.transition_w_values.items():
            canonical.transition_w_values.setdefault((i, w), out)
        for i, learned in duplicate.learned_w_per_input.items():
            canonical.learned_w_per_input[i].update(learned)
        for i, out in duplicate.output_fun.items():
            canonical.output_fun.setdefault(i, out)
        for i, trans in duplicate.transitions.items():
            canonical.transitions.setdefault(i, trans)

        for state in self._all_states():
            for i, target in list(state.transitions.items()):
                if target is duplicate:
                    state.transitions[i] = canonical

        for h_response, state in list(self.h_response_map.items()):
            if state is duplicate:
                self.h_response_map[h_response] = canonical

    def _complete_h_response_state(self, h_response, state):
        profile = self._w_profile(state.state_w_values)
        existing = self.state_map.get(profile)

        if existing is None:
            state.hs = profile
            self.state_map[profile] = state
            self.h_response_map[h_response] = state
            return state

        if existing is not state:
            self._merge_states(existing, state)
            self.h_response_map[h_response] = existing
        return existing

    def update_model_transitions(self):
        current_w_set = set(self.W)
        new_states = {}

        for hs, state in self.state_map.items():
            # Pre-group transition_w_values by input in one pass instead of
            # re-scanning the full dict for every input symbol.
            by_input = defaultdict(dict)
            for (i, w), output in state.transition_w_values.items():
                if w in current_w_set:
                    by_input[i][w] = output

            for i in self.input_alphabet:
                w_for_input = by_input.get(i, {})
                if len(w_for_input) != len(self.W):
                    continue

                matched = self._state_for_w_values(w_for_input, new_states)
                if matched is None:
                    continue
                state.transitions[i] = matched
        self.state_map.update(new_states)

    def update_model_transition(self, state, i):
        if len(state.learned_w_per_input[i]) != len(self.W):
            return False

        w_for_input = {
            w: output
            for (ii, w), output in state.transition_w_values.items()
            if ii == i and w in state.learned_w_per_input[i]
        }
        matched = self._state_for_w_values(w_for_input)
        if matched is None:
            return False

        state.transitions[i] = matched
        return False

    def _state_partitions(self):
        states = [s for s in self.state_map.values() if len(s.state_w_values) == len(self.W)]

        block_of = {}
        output_blocks = {}
        for state in states:
            signature = tuple(state.output_fun.get(i) for i in self.input_alphabet)
            output_blocks.setdefault(signature, len(output_blocks))
            block_of[state.hs] = output_blocks[signature]

        changed = True
        while changed:
            changed = False
            next_blocks = {}
            next_block_of = {}
            for state in states:
                signature = tuple(
                    (
                        i,
                        state.output_fun.get(i),
                        block_of.get(state.transitions[i].hs)
                        if i in state.transitions and isinstance(state.transitions[i], ModelState)
                        else None
                    )
                    for i in self.input_alphabet
                )
                next_blocks.setdefault(signature, len(next_blocks))
                next_block_of[state.hs] = next_blocks[signature]
            if next_block_of != block_of:
                changed = True
                block_of = next_block_of

        return block_of

    def _model_state_h_output(self, state):
        outputs = []
        for i in self.homing_sequence:
            if i not in state.output_fun or i not in state.transitions:
                return None
            outputs.append(state.output_fun[i])
            state = state.transitions[i]
        return tuple(outputs)

    def create_model(self, current_hs):
        block_of = self._state_partitions()
        block_representatives = {}
        for hs, block in block_of.items():
            block_representatives.setdefault(block, hs)

        block_states = {}
        automata_states = {}
        for block, hs in block_representatives.items():
            mealy_state = MealyState(f's{len(block_states)}')
            mealy_state.prefix = hs
            block_states[block] = mealy_state
            automata_states[hs] = mealy_state

        for block, hs in block_representatives.items():
            state = self.state_map[hs]
            automata_state = block_states[block]
            for i in self.input_alphabet:
                if i in state.transitions and isinstance(state.transitions[i], ModelState):
                    target_block = block_of.get(state.transitions[i].hs)
                    if target_block is not None:
                        automata_state.transitions[i] = block_states[target_block]
                        automata_state.output_fun[i] = state.output_fun[i]

        start_state = automata_states.get(current_hs)
        if start_state is None:
            start_state = next(iter(block_states.values()))

        reachable_states = []
        queue = deque([start_state])
        seen = set()
        while queue:
            state = queue.popleft()
            if state in seen:
                continue
            seen.add(state)
            reachable_states.append(state)
            for next_state in state.transitions.values():
                queue.append(next_state)

        for s in reachable_states:
            for i in self.input_alphabet:
                if i not in s.transitions:
                    s.transitions[i] = s
                    s.output_fun.setdefault(i, None)

        mm = MealyMachine(start_state, reachable_states)
        mm.current_state = start_state
        return mm

    def create_hypothesis(self):
        while True:
            hs_response = self.execute_homing_sequence()

            if self.interrupt:
                self.interrupt = False
                continue

            current_state = self.h_response_map.get(hs_response)
            if current_state is None:
                current_state = ModelState(hs_response)
                self.h_response_map[hs_response] = current_state

            if len(current_state.state_w_values) != len(self.W):
                for w in self.W:
                    if w not in current_state.state_w_values:
                        dict_key = (self.homing_sequence, hs_response, w)
                        w_response = self.h_w_dictionary.get(dict_key)
                        if w_response is None:
                            w_response = self.execute_sequence(w)
                            if not self.interrupt:
                                self.h_w_dictionary[dict_key] = w_response
                        if not self.interrupt:
                            current_state.state_w_values[w] = w_response
                        break
                if not self.interrupt and len(current_state.state_w_values) == len(self.W):
                    current_state = self._complete_h_response_state(hs_response, current_state)
                    if not self.check_w_ND_from_state_map():
                        self.interrupt = False
                        continue
            else:
                reachable_incomplete = self.find_reachable_incomplete_transition(current_state)
                if reachable_incomplete is None:
                    if self.check_conjecture_inconsistencies(current_state):
                        if self.interrupt:
                            self.interrupt = False
                        continue
                    if self.is_complete():
                        break
                    return self.create_model(current_state.hs)

                alpha, target, x, w = reachable_incomplete

                dict_key = (self.homing_sequence, hs_response, alpha, x, w)
                cached = self.dictionary.get(dict_key)
                if cached is None:
                    alpha_result = self.execute_conjecture_path(current_state, alpha)
                    if self.interrupt:
                        self.interrupt = False
                        continue
                    if alpha_result is None:
                        continue

                    reached_after_alpha, alpha_response = alpha_result
                    if reached_after_alpha is not target:
                        target = reached_after_alpha

                    output = self.step_wrapper(x)
                    w_response = self.execute_sequence(w)

                    if not self.interrupt:
                        self.dictionary[dict_key] = (alpha_response, output, w_response)
                else:
                    _, output, w_response = cached

                if self.interrupt:
                    self.interrupt = False
                    continue

                target.transition_w_values[(x, w)] = w_response
                target.learned_w_per_input[x].add(w)
                target.output_fun[x] = output

                self.update_model_transition(target, x)

        hypothesis = self.create_model(current_state.hs)
        return hypothesis

    def main_loop(self, print_level=2):
        start_time = time.time()
        eq_query_time = 0

        initial_model = self.create_daisy_hypothesis()

        eq_start = time.time()
        counter_example = self.find_counterexample(initial_model)
        eq_query_time += time.time() - eq_start

        last_cex_input = (counter_example[-1],)

        self.homing_sequence = last_cex_input
        self.add_to_W(last_cex_input)

        self.sul.h = self.homing_sequence

        learning_rounds = 0
        while True:
            learning_rounds += 1

            hypothesis = self.create_hypothesis()

            if print_level > 1:
                print(f'Hypothesis {learning_rounds}: {hypothesis.size} states.')

            eq_start = time.time()
            counter_example = self.find_counterexample(hypothesis)
            eq_query_time += time.time() - eq_start

            if counter_example is None:
                break

            cex_suffixes = sorted((tuple(s) for s in all_suffixes(counter_example)), key=len)
            added_suffix = None

            for s in cex_suffixes:
                if self.add_to_W(s):
                    added_suffix = s
                    break

            if added_suffix is None:
                full_cex = tuple(counter_example)
                added_suffix = self.homing_sequence + full_cex
                if not self.add_to_W(added_suffix):
                    break

            # Ensure h is always in W
            self.add_h_to_W()

            self._reset_state_data()
            self.interrupt = False


        # finalize initial reset cleanup
        self.sul.post()

        queries_before_initial_state = self.sul.num_queries

        if self.query_for_initial_state:
            initial_w_values = {}
            for w in self.W:
                initial_w_values[w] = tuple(self.sul.query(w))

            initial_hs_response = tuple(self.sul.query(self.homing_sequence))
            for r, model_state in self.state_map.items():
                if model_state.state_w_values == initial_w_values:
                    canonical = model_state.hs  # use canonical key after rekeying
                    for s in hypothesis.states:
                        if s.prefix == canonical or canonical in getattr(s, 'prefixes', set()):
                            hypothesis.initial_state = s
                            break
                    break
            else:
                for r, model_state in self.state_map.items():
                    if self._model_state_h_output(model_state) == initial_hs_response:
                        for s in hypothesis.states:
                            if s.prefix == r or r in getattr(s, 'prefixes', set()):
                                hypothesis.initial_state = s
                                break
                        break

        total_time = round(time.time() - start_time, 2)
        eq_query_time = round(eq_query_time, 2)
        learning_time = round(total_time - eq_query_time, 2)

        info = {
            'learning_rounds': learning_rounds,
            'automaton_size': hypothesis.size,
            'queries_learning': self.sul.num_queries,
            'queries_eq_oracle': 0, # oracle does not reset
            'steps_learning': self.sul.num_steps,
            'steps_eq_oracle': self.total_testing_steps,
            'learning_time': learning_time,
            'eq_oracle_time': eq_query_time,
            'total_time': total_time,
            'homing_sequence': self.homing_sequence,
            'characterization_set': self.W,
        }

        if self.query_for_initial_state:
            info['queries_initial_state'] = self.sul.num_queries - queries_before_initial_state

        if print_level > 0:
            print_learning_info(info)

        return hypothesis, info


def run_hW(alphabet: list, sul, num_testing_steps=200, reset_testing_counter=True,
           query_for_initial_state=True, return_data=False, print_level=2):
    """
    Executes the hW resetless learning algorithm.
    Algorithm description can be found in "hW-inference: A heuristic approach to retrieve models through
    black box testing" by Groz et al.

    The implementation does not strictly follow all aspects of the described algorithm, but relies on in for the most
    part.

    Args:

        alphabet: input alphabet

        sul: system under learning

        num_testing_steps: number of random steps used per equivalence check (Default value = 100)

        reset_testing_counter: reset the testing step counter after each counterexample (Default value = True)

        query_for_initial_state: if True, query the SUL to identify the true initial state (Default value = True)

        return_data: if True, return a (hypothesis, info) tuple instead of just the hypothesis
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

    Returns:

        learned Mealy machine (or (machine, info dict) if return_data is True)
    """
    assert print_level in [0, 1, 2, 3]

    hw = hW(
        input_al=alphabet,
        sul=sul,
        num_testing_steps=num_testing_steps,
        reset_testing_counter=reset_testing_counter,
        query_for_initial_state=query_for_initial_state,
    )

    hypothesis, info = hw.main_loop(print_level=print_level)

    if return_data:
        return hypothesis, info

    return hypothesis
