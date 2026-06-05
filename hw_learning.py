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
        self.transition_target_key = {}  # input -> hs_response of post-h target state


class PendingState:
    def __init__(self, hs, state_w_values):
        self.hs = hs
        self.state_w_values = dict(state_w_values)


class hW:
    def __init__(self, input_al, sul,
                 num_testing_steps=100,
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

        self.state_map = {}
        self._states_by_w_profile = {}
        self._ambiguous_w_profiles = set()

        # Incremental h-ND index
        self._hs_cont_starts = defaultdict(list)   # hs_output -> [cont_start_pos, ...] (ordered for determinism)
        self._pair_progress = {}                   # (p1, p2) -> steps compared
        self._h_nd_scan_pos = 0                    # how far global_trace has been scanned

        self._pending_pool = {}                    # w_profile -> PendingState (dedup to prevent state explosion)

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

    def execute_homing_sequence(self):
        observed_output = [self.step_wrapper(i) for i in self.homing_sequence]
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

    def _reset_state_data(self):
        self.state_map.clear()
        self._states_by_w_profile.clear()
        self._ambiguous_w_profiles.clear()
        self._pending_pool.clear()
        self._reset_h_nd_index()

    def _is_profile_key(self, key):
        return isinstance(key, tuple) and key and isinstance(key[0], tuple)

    def _refine_w_if_profile_explosion(self):
        profile_keys = sum(1 for key in self.state_map if self._is_profile_key(key))

        real_keys = max(1, len(self.state_map) - profile_keys)
        if profile_keys <= real_keys:
            return False

        for symbol in reversed(self.input_alphabet):
            suffix = (symbol,)
            if suffix not in self.W:
                self.add_to_W(suffix)
                self._reset_state_data()
                self.interrupt = True
                return True

        return False

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
                    self.add_to_W(self.homing_sequence, preserve_h=False)
                    return False
            else:
                if compare_len > already:
                    self._pair_progress[(p1, p2)] = compare_len

        for pair in to_delete:
            del self._pair_progress[pair]

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
        queue = deque([(start_state, (), None, None)])
        visited = set()
        pending_candidates = []

        while True:
            while queue:
                state, path, parent, parent_input = queue.popleft()
                if isinstance(state, PendingState):
                    pending_candidates.append((state, path, parent, parent_input))
                    continue

                if state in visited:
                    continue

                visited.add(state)

                missing = self._first_missing_transition_query(state)
                if missing is not None:
                    x, w = missing
                    return path, state.hs, x, w

                for input_sequence, next_state in state.transitions.items():
                    queue.append((next_state, path + (input_sequence,), state, input_sequence))

            if not pending_candidates:
                return None

            pending, path, parent, parent_input = pending_candidates.pop(0)
            state = self._materialize_pending_state(pending)
            if parent is not None:
                parent.transitions[parent_input] = state
            queue.append((state, path, parent, parent_input))

    def _w_profile(self, w_values):
        return tuple(sorted(w_values.items()))

    def _register_w_profile(self, state):
        if len(state.state_w_values) != len(self.W):
            return

        profile = self._w_profile(state.state_w_values)
        if profile in self._ambiguous_w_profiles:
            return

        existing = self._states_by_w_profile.get(profile)
        if existing is None:
            self._states_by_w_profile[profile] = state
        elif existing is not state:
            self._states_by_w_profile.pop(profile, None)
            self._ambiguous_w_profiles.add(profile)

    def _find_state_by_w_values(self, w_values, new_states=None):
        profile = self._w_profile(w_values)
        if profile not in self._ambiguous_w_profiles:
            matched = self._states_by_w_profile.get(profile)
            if matched is not None:
                return matched

        for destination_state in self.state_map.values():
            if w_values == destination_state.state_w_values:
                return destination_state

        if new_states:
            for destination_state in new_states.values():
                if w_values == destination_state.state_w_values:
                    return destination_state

        return None

    def _is_consistent_profile(self, known_values, w_values):
        return all(
            w_values.get(k) == v
            for k, v in known_values.items()
            if k in w_values
        )

    def _find_consistent_partial_match(self, w_values, new_states=None):
        """Return the unique state whose known w-values are a consistent (non-empty)
        subset of w_values, or None if there is no such state or the match is ambiguous.
        When a unique match is found its state_w_values are updated with the new values."""
        partial_match = None
        candidates = self.state_map.values()
        if new_states:
            candidates = list(candidates) + list(new_states.values())

        for state in candidates:
            if not state.state_w_values:
                continue
            if self._is_consistent_profile(state.state_w_values, w_values):
                if partial_match is not None:
                    return None  # ambiguous
                partial_match = state
        if partial_match is not None:
            partial_match.state_w_values.update(w_values)
            if len(partial_match.state_w_values) == len(self.W):
                self._register_w_profile(partial_match)
        return partial_match

    def _key_from_w_values(self, w_values, existing_states=None):
        h = self.homing_sequence
        if h in w_values:
            new_key = w_values[h]
        else:
            candidates = [w for w in self.W if len(w) >= len(h) and w[:len(h)] == h]
            if not candidates:
                return None
            new_key = w_values[candidates[0]][:len(h)]

        existing = self.state_map.get(new_key)
        if existing is None and existing_states:
            existing = existing_states.get(new_key)
        if existing is not None and existing.state_w_values != w_values:
            conflicts = any(
                existing.state_w_values.get(k) != v
                for k, v in w_values.items()
                if k in existing.state_w_values
            )
            if conflicts:
                new_key = tuple(sorted(w_values.items()))
        return new_key

    def _pending_or_existing_state(self, w_values, new_states=None):
        matched = self._find_state_by_w_values(w_values, new_states)
        if matched is not None:
            return matched

        new_key = self._key_from_w_values(w_values, new_states)
        if new_key is None:
            return None

        matched = self.state_map.get(new_key)
        if matched is None and new_states:
            matched = new_states.get(new_key)
        if matched is not None:
            return matched

        profile = self._w_profile(w_values)
        pending = self._pending_pool.get(profile)
        if pending is None:
            pending = PendingState(new_key, w_values)
            self._pending_pool[profile] = pending
        return pending

    def _materialize_pending_state(self, pending):
        # pending.hs is already a full W-profile key (set by _key_from_w_values).
        # Check for an exact match first (handles the case where the same state
        # was rekeyed or materialized earlier under the same full-profile key).
        matched = self._find_state_by_w_values(pending.state_w_values)
        if matched is not None:
            return matched

        existing = self.state_map.get(pending.hs)
        if existing is not None:
            return existing

        state = ModelState(pending.hs)
        state.state_w_values = dict(pending.state_w_values)
        self.state_map[state.hs] = state
        self._register_w_profile(state)
        return state

    def _rekey_completed_state(self, old_key, state):
        """Move a fully-W-queried state from its temporary homing-sequence-response
        key to its stable full-W-profile key.

        If a transition-derived state with that full-profile key already exists
        (materialized from a PendingState before this direct visit), merge the two
        into one, transferring all learned transition data and redirecting all
        pointers.  Keep the old h-response key as an alias so the outer loop can
        still find the state by its homing response without re-querying W.
        """
        new_key = tuple(sorted(state.state_w_values.items()))
        if new_key == old_key:
            return state

        existing = self.state_map.get(new_key)
        # Add new_key as an additional alias (keeps old_key working too).
        self.state_map[new_key] = state
        state.hs = new_key

        if existing is not None and existing is not state:
            # Transfer learned transition data from the transition-derived state
            # into the now-complete directly-visited state.
            for (i, w), out in existing.transition_w_values.items():
                if (i, w) not in state.transition_w_values:
                    state.transition_w_values[(i, w)] = out
            for i, learned in existing.learned_w_per_input.items():
                state.learned_w_per_input[i].update(learned)
            for i, out in existing.output_fun.items():
                if i not in state.output_fun:
                    state.output_fun[i] = out
            for i, trans in existing.transitions.items():
                if i not in state.transitions:
                    state.transitions[i] = trans
            # Redirect any transitions pointing to the old duplicate → canonical.
            for s in self.state_map.values():
                for i, t in list(s.transitions.items()):
                    if t is existing:
                        s.transitions[i] = state
            # Remove the now-orphaned duplicate entry (find it by identity).
            for k, v in list(self.state_map.items()):
                if v is existing and k != new_key:
                    del self.state_map[k]

        return state

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

                matched = self._pending_or_existing_state(w_for_input, new_states)
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
        matched = self._pending_or_existing_state(w_for_input)
        if matched is None:
            return False

        state.transitions[i] = matched
        return False

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
                if i in state.transitions and isinstance(state.transitions[i], ModelState):
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
                    self._register_w_profile(current_state)
                    if not self.check_w_ND_from_state_map():
                        self.interrupt = False
                        continue
            else:
                reachable_incomplete = self.find_reachable_incomplete_transition(current_state)
                if reachable_incomplete is None:
                    if self.is_complete():
                        break
                    return self.create_model(hs_response)

                alpha, reached_state, x, w = reachable_incomplete

                self.execute_sequence(alpha)
                output = self.step_wrapper(x)
                w_response = self.execute_sequence(w)

                if self.interrupt:
                    self.interrupt = False
                    continue

                target = self.state_map[reached_state]
                target.transition_w_values[(x, w)] = w_response
                target.learned_w_per_input[x].add(w)
                target.output_fun[x] = output

                self.update_model_transition(target, x)

        hypothesis = self.create_model(hs_response)
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
            self.add_to_W(self.homing_sequence)

            self._reset_state_data()
            self.interrupt = False

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


def run_HW(alphabet: list, sul, num_testing_steps=100, reset_testing_counter=True,
           query_for_initial_state=False, return_data=False, print_level=2):
    """
    Executes the hW (homing-sequence Witness) learning algorithm.

    Args:

        alphabet: input alphabet

        sul: system under learning

        num_testing_steps: number of random steps used per equivalence check (Default value = 100)

        reset_testing_counter: reset the testing step counter after each counterexample (Default value = True)

        query_for_initial_state: if True, query the SUL to identify the true initial state (Default value = False)

        max_learning_rounds: stop after these many rounds if set (Default value = None)

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
