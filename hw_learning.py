from collections import defaultdict, deque
from random import choice

from aalpy.automata import MealyState, MealyMachine
from aalpy.utils.HelperFunctions import all_suffixes


class ModelState:
    def __init__(self, hs):
        self.hs = hs
        self.state_w_values = dict()

        self.transitions = dict()
        self.output_fun = dict()

        self.transition_w_values = dict()

    def get_paths_to_reachable_states(self, target_states):
        target_states = set(target_states)
        if self.hs in target_states:
            return [((), self.hs)]

        queue = deque([(self, ())])
        visited = set()

        while queue:
            current_node, path_taken = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)

            for input_sequence, next_node in current_node.transitions.items():
                next_path = path_taken + (input_sequence,)
                if next_node.hs in target_states:
                    return [(next_path, next_node.hs)]
                queue.append((next_node, next_path))

        return []


class hW:
    def __init__(self, input_al, sul,
                 num_testing_steps=100,
                 reset_testing_counter=True,
                 query_for_initial_state=False):

        self.input_alphabet = input_al
        self.sul = sul
        self.total_testing_steps = 0
        self.num_testing_steps = num_testing_steps
        self.reset_testing_counter = reset_testing_counter
        self.query_for_initial_state = query_for_initial_state

        self.homing_sequence = ()
        self.W = []

        self.global_trace = []

        self.interrupt = False

        self.state_map = dict()

        # Incremental h-ND index
        self._hs_cont_starts = defaultdict(set)    # hs_output -> {cont_start_pos, ...}
        self._pair_progress = {}                   # (p1, p2) -> steps compared (-1 = inputs diverged)
        self._h_nd_scan_pos = 0                    # how far global_trace has been scanned

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
            or not (w != sequence and sequence[:len(w)] == w)
        ]
        return True

    def execute_homing_sequence(self):
        observed_output = []

        for i in self.homing_sequence:
            output = self.sul.step(i)
            if isinstance(output, tuple) and len(output) == 1:
                output = output[0]
            self.global_trace.append((i, output))
            self.sul.num_steps += 1
            observed_output.append(output)

        self.check_h_ND_consistency()
        return tuple(observed_output)

    def execute_sequence(self, seq_under_test: tuple):
        observed_output = []
        for i in seq_under_test:
            observed_output.append(self.step_wrapper(i))
        return tuple(observed_output)

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

    def check_h_ND_consistency(self):
        h_len = len(self.homing_sequence)
        if h_len == 0:
            return True

        # Incrementally scan only newly added trace positions for new hs occurrences
        scan_start = max(0, self._h_nd_scan_pos - h_len + 1)
        for i in range(scan_start, len(self.global_trace) - h_len + 1):
            prefix_inputs = tuple(inp for inp, _ in self.global_trace[i:i + h_len])
            if prefix_inputs == self.homing_sequence:
                prefix_outputs = tuple(out for _, out in self.global_trace[i:i + h_len])
                cont_start = i + h_len
                self._hs_cont_starts[prefix_outputs].add(cont_start)
        self._h_nd_scan_pos = len(self.global_trace)

        # Incrementally compare pairs — each pair advances only from where it last stopped
        for prefix_outputs, cont_starts in self._hs_cont_starts.items():
            if len(cont_starts) <= 1:
                continue
            cont_starts = tuple(sorted(cont_starts))
            for idx_i in range(len(cont_starts)):
                for idx_j in range(idx_i + 1, len(cont_starts)):
                    p1, p2 = cont_starts[idx_i], cont_starts[idx_j]
                    already = self._pair_progress.get((p1, p2), 0)
                    if already == -1:
                        continue  # inputs already diverged for this pair

                    compare_len = min(len(self.global_trace) - p1, len(self.global_trace) - p2)
                    for k in range(already, compare_len):
                        inp1, out1 = self.global_trace[p1 + k]
                        inp2, out2 = self.global_trace[p2 + k]
                        if inp1 != inp2:
                            self._pair_progress[(p1, p2)] = -1
                            break
                        if out1 != out2:
                            ext = tuple(inp for inp, _ in self.global_trace[p1:p1 + k + 1])
                            self.homing_sequence += ext
                            self.interrupt = True
                            self.state_map.clear()
                            self._reset_h_nd_index()

                            self.add_to_W(self.homing_sequence, preserve_h=False)

                            return False
                        self._pair_progress[(p1, p2)] = k + 1

        return True

    def check_w_ND_from_state_map(self):
        states = list(self.state_map.items())
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
                                print(f'  [w-ND] {hs1} ≡ {hs2} under W — adding w={new_w}  W={self.W + [new_w]}')
                                self.add_to_W(new_w)
                                self.state_map.clear()
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

    def _find_ce_via_dummy(self, hypothesis, start_state=None):
        start_state = start_state or hypothesis.initial_state or hypothesis.current_state
        queue = deque([(start_state, [])])
        visited = set()

        while queue:
            state, path = queue.popleft()
            if state in visited:
                continue
            visited.add(state)

            for i in self.input_alphabet:
                if state.output_fun.get(i) == 'dummy_output':
                    return path + [i]
                next_state = state.transitions.get(i)
                if next_state is not None:
                    queue.append((next_state, path + [i]))
        return None

    def find_counterexample(self, hypothesis):
        dummy_cex = self._find_ce_via_dummy(hypothesis)
        if dummy_cex is not None:
            return dummy_cex

        cex = []
        for _ in range(self.num_testing_steps):
            self.total_testing_steps += 1
            random_input = choice(self.input_alphabet)
            cex.append(random_input)
            o_sul = self.step_wrapper(random_input)
            o_hyp = hypothesis.step(random_input)
            if o_sul != o_hyp:
                if self.reset_testing_counter:
                    self.total_testing_steps = 0

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
                if state.transitions[i].hs not in self.state_map:
                    return False
        return True

    def get_incomplete_transitions(self):
        incomplete_transitions = defaultdict(list)
        for hs, state in self.state_map.items():
            learned_w_per_input = defaultdict(set)
            for (i, w) in state.transition_w_values.keys():
                learned_w_per_input[i].add(w)
            for i in self.input_alphabet:
                missing_ws = [w for w in self.W if w not in learned_w_per_input[i]]
                if missing_ws:
                    incomplete_transitions[hs].append((i, missing_ws[0]))
        return incomplete_transitions

    def _profile_key(self, base_key, w_values):
        return ('profile', base_key, tuple(sorted(w_values.items())))

    def update_model_transitions(self):
        current_w_set = set(self.W)
        new_states = {}

        for hs, state in self.state_map.items():
            for i in self.input_alphabet:
                w_for_input = {w: output for (ii, w), output in state.transition_w_values.items()
                               if ii == i and w in current_w_set}
                if len(w_for_input) == len(self.W):
                    matched = None
                    for _, destination_state in self.state_map.items():
                        if w_for_input == destination_state.state_w_values:
                            matched = destination_state
                            break
                    if matched is None:
                        for _, destination_state in new_states.items():
                            if w_for_input == destination_state.state_w_values:
                                matched = destination_state
                                break
                    if matched is None:
                        h = self.homing_sequence
                        if h in w_for_input:
                            new_key = w_for_input[h]
                        else:
                            candidates = [w for w in self.W if len(w) >= len(h) and w[:len(h)] == h]
                            if not candidates:
                                continue
                            new_key = w_for_input[candidates[0]][:len(h)]
                        existing = self.state_map.get(new_key) or new_states.get(new_key)
                        if existing is not None and existing.state_w_values != w_for_input:
                            new_key = self._profile_key(new_key, w_for_input)
                        if new_key not in new_states and new_key not in self.state_map:
                            new_state = ModelState(new_key)
                            new_state.state_w_values = dict(w_for_input)
                            new_states[new_key] = new_state
                        matched = self.state_map.get(new_key) or new_states.get(new_key)
                        if matched is None:
                            continue
                    state.transitions[i] = matched
        self.state_map.update(new_states)

    def _state_partitions(self):
        states = list(self.state_map.values())
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
                        block_of.get(state.transitions[i].hs) if i in state.transitions else None
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

        automata_states = {}
        block_states = {}
        for block, hs in block_representatives.items():
            mealy_state = MealyState(f's{len(block_states)}')
            mealy_state.prefix = hs
            mealy_state.prefixes = {k for k, b in block_of.items() if b == block}
            block_states[block] = mealy_state
            for prefix in mealy_state.prefixes:
                automata_states[prefix] = mealy_state

        wired_blocks = set()
        for hs, state in self.state_map.items():
            block = block_of[hs]
            if block in wired_blocks:
                continue
            wired_blocks.add(block)
            automata_state = block_states[block]
            for i in self.input_alphabet:
                if i not in state.transitions.keys():
                    automata_state.transitions[i] = automata_state
                    automata_state.output_fun[i] = 'dummy_output'
                else:
                    automata_state.transitions[i] = automata_states[state.transitions[i].hs]
                    automata_state.output_fun[i] = state.output_fun[i]

        start_state = automata_states[current_hs]
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

        mm = MealyMachine(start_state, reachable_states)
        mm.current_state = start_state
        dummy_count = sum(1 for s in reachable_states for o in s.output_fun.values() if o == 'dummy_output')
        print(f'  [create_model] {len(reachable_states)} states, {dummy_count} dummy transitions, current={current_hs}')
        return mm

    def create_hypothesis(self):
        while True:
            hs_response = self.execute_homing_sequence()

            if self.interrupt:
                self.interrupt = False
                continue

            if hs_response not in self.state_map:
                self.state_map[hs_response] = ModelState(hs_response)

            current_state = self.state_map[hs_response]

            if len(current_state.state_w_values) != len(self.W):
                for w in self.W:
                    if w not in current_state.state_w_values:
                        w_response = self.execute_sequence(w)
                        if not self.interrupt:
                            current_state.state_w_values[w] = w_response
                        break
                if not self.interrupt and len(current_state.state_w_values) == len(self.W):
                    if not self.check_w_ND_from_state_map():
                        self.interrupt = False
                        continue
            else:
                complete = sum(1 for s in self.state_map.values() if len(s.state_w_values) == len(self.W))
                print(f'  [create_hypothesis] states={len(self.state_map)} ({complete} complete)  hs={hs_response}  W={self.W}')

                tail_node = self.state_map[hs_response]
                incomplete_transitions = self.get_incomplete_transitions()
                incomplete_states = list(incomplete_transitions.keys())

                if not incomplete_states:
                    if self.is_complete():
                        print(f'  [create_hypothesis] complete — exiting')
                        break
                    else:
                        continue

                paths_to_reachable_states = tail_node.get_paths_to_reachable_states(incomplete_states)

                if not paths_to_reachable_states:
                    if self.is_complete():
                        print(f'  [create_hypothesis] complete — exiting')
                        break
                    print(f'  [create_hypothesis] no path to incomplete states {incomplete_states} — returning partial model')
                    return self.create_model(hs_response)

                alpha, reached_state = paths_to_reachable_states[0]
                x = incomplete_transitions[reached_state][0][0]
                w = incomplete_transitions[reached_state][0][1]

                self.execute_sequence(alpha)
                output = self.step_wrapper(x)
                w_response = self.execute_sequence(w)

                if self.interrupt:
                    self.interrupt = False
                    continue

                self.state_map[reached_state].transition_w_values[(x, w)] = w_response
                self.state_map[reached_state].output_fun[x] = output

                self.update_model_transitions()

        hypothesis = self.create_model(hs_response)
        return hypothesis

    def main_loop(self):
        initial_model = self.create_daisy_hypothesis()

        counter_example = self.find_counterexample(initial_model)
        last_cex_input = tuple([counter_example[-1]])

        self.homing_sequence = last_cex_input
        self.add_to_W(last_cex_input)

        self.sul.h = self.homing_sequence

        iteration = 0
        while True:
            iteration += 1
            print(f'[main_loop] iter={iteration}  h={self.homing_sequence}  W={self.W}')
            hypothesis = self.create_hypothesis()
            counter_example = self.find_counterexample(hypothesis)

            if counter_example is None:
                print(f'[main_loop] no counterexample — done')
                break

            cex_suffixes = sorted([tuple(s) for s in all_suffixes(counter_example)], key=len)
            added_suffix = None

            for s in cex_suffixes:
                if self.add_to_W(s):
                    added_suffix = s
                    break

            if added_suffix is None:
                full_cex = tuple(counter_example)
                added_suffix = self.homing_sequence + full_cex
                if not self.add_to_W(added_suffix):
                    print(f'  [main_loop] all suffixes already in W — stopping without duplicating {full_cex}')
                    break

            # Ensure h is always in W
            self.add_to_W(self.homing_sequence)

            print(f'[main_loop] cex={counter_example}  +suffix={added_suffix}  W={self.W}')
            self.state_map.clear()
            self._reset_h_nd_index()
            self.interrupt = False

        if self.query_for_initial_state:
            initial_w_values = {}
            for w in self.W:
                self.sul.num_queries += 1
                initial_w_values[w] = tuple(self.sul.query(w))

            initial_hs_response = tuple(self.sul.query(self.homing_sequence))
            for r, model_state in self.state_map.items():
                if model_state.state_w_values == initial_w_values:
                    for s in hypothesis.states:
                        if s.prefix == r or r in getattr(s, 'prefixes', set()):
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

        print(f'[main_loop] learned {hypothesis.size} states  resets={self.sul.num_queries}  steps={self.sul.num_steps}')

        return hypothesis
