import time
from collections import defaultdict, deque
from random import choice

from aalpy.automata import MealyState, MealyMachine, MooreState, MooreMachine, DfaState, Dfa
from aalpy.utils.HelperFunctions import all_suffixes, print_learning_info


class ModelState:
    """
    A state of the conjecture under construction.
    """

    def __init__(self, hs):
        self.hs = hs
        self.state_w_values = {}

        self.transitions = {}
        self.output_fun = {}

        self.transition_w_values = {}
        self.learned_w_per_input = defaultdict(set)


class hW:
    """Adaptation of hW resetless inference (Groz et al.) capable of learning Mealy and Moore machines, as well as DFAs.

    States are identified by their responses to the characterization set W after an
    application of the homing sequence h; both h and W are refined on the fly.

    Optimizations:
    - all query answers are cached, and answers are additionally mined from the
      already-observed trace (_mine_h_w_response), so nothing is asked twice
    - homing is skipped whenever the SUL's current state is still known, e.g. after
      cache hits or probes fully predicted by the conjecture (tracked_state)
    - h non-determinism is checked incrementally over the global trace, skipping
      self-overlapping h occurrences that would blow up the index quadratically
    """

    def __init__(self, input_al, sul,
                 automaton_type='mealy',
                 num_testing_steps=200,
                 reset_testing_counter=True,
                 query_for_initial_state=False,
                 H=None,
                 W=None):

        assert automaton_type in ('mealy', 'moore')
        self.is_moore = automaton_type == 'moore'
        self.input_alphabet = input_al # to ensure determinism if passed as set
        self.sul = sul
        self.total_testing_steps = 0
        self.num_testing_steps = num_testing_steps
        self.reset_testing_counter = reset_testing_counter
        self.query_for_initial_state = query_for_initial_state

        self.homing_sequence = tuple(H) if H is not None else ()
        self.W = []
        self._user_provided_h = H is not None

        self.global_trace = []

        self.interrupt = False

        self.state_map = {}                # full W-profile -> ModelState
        self.h_response_map = {}           # h-response -> partial or complete ModelState
        self.dictionary = {}               # (h, h_response, alpha, x, w) -> (alpha_out, x_out, w_out)
        self.h_w_dictionary = {}           # (h, h_response, w) -> w_out
        self._conjecture_probe_seen = set()
        self._aut_state_by_hs = {}         # hs -> automaton state of the latest created model

        # incremental h-ND index over global_trace
        self._hs_cont_starts = defaultdict(list)   # h-response -> [continuation start positions]
        self._hs_cont_set = set()                  # all registered continuation starts (O(1) membership)
        self._pair_progress = {}                   # (p1, p2) -> compared continuation length so far
        self._h_nd_scan_pos = 0                    # how far global_trace has been scanned
        self._next_occ_min_start = 0               # earliest start of the next registered h occurrence

        # reset/initialize the SUL
        sul.pre()

        if W is not None:
            for w in W:
                self.add_to_W(tuple(w))
        if self.is_moore:
            self.add_to_W(())
        if self._user_provided_h:
            self.add_h_to_W()

    @staticmethod
    def _unwrap_output(output):
        """Unpack 1-element output tuples returned by some SULs."""
        if isinstance(output, tuple) and len(output) == 1:
            return output[0]
        return output

    def add_to_W(self, sequence, preserve_h=True):
        """Add a sequence to W and drop its proper prefixes (h and the Moore empty
        suffix are always kept). Returns True if W changed."""
        sequence = tuple(sequence)
        if sequence in self.W:
            return False

        self.W.append(sequence)
        self.W = [
            w for w in self.W
            if w == sequence
            or (preserve_h and w == self.homing_sequence)
            or (self.is_moore and w == ())
            or sequence[:len(w)] != w
        ]
        return True

    def add_h_to_W(self):
        """Ensure h (or an extension of it) is part of W."""
        h = self.homing_sequence
        if h in self.W:
            return False
        if any(w[:len(h)] == h for w in self.W):
            return False
        return self.add_to_W(h, preserve_h=False)

    def step_wrapper(self, letter):
        """Single SUL step that is recorded in the global trace."""
        output = self._unwrap_output(self.sul.step(letter))
        self.global_trace.append((letter, output))
        self.sul.num_steps += 1
        return output

    def execute_homing_sequence(self):
        """Execute h, run the non-determinism check, and return the observed response."""
        response = tuple(self.step_wrapper(i) for i in self.homing_sequence)
        self.check_h_ND_consistency()
        return response

    def execute_sequence(self, seq_under_test: tuple):
        """Execute a sequence and return its outputs. The empty sequence (Moore only)
        reads the current state output without moving the SUL."""
        if not seq_under_test and self.is_moore:
            return (self._unwrap_output(self.sul.step(None)),)
        return tuple(self.step_wrapper(i) for i in seq_under_test)

    def execute_conjecture_path(self, start_state, path):
        """Walk path on the SUL while verifying outputs against the conjecture.
        On a mismatch, extend W with the failing prefix (or drop the stale transition
        data if the prefix is already in W) and return None.
        Returns (reached state, observed outputs) on success."""
        state = start_state
        observed_outputs = []
        for pos, i in enumerate(path):
            observed_output = self.step_wrapper(i)
            observed_outputs.append(observed_output)
            if state.output_fun.get(i) != observed_output:
                if self.add_to_W(path[:pos + 1]):
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
        """Single-state hypothesis with self-loops; its first counterexample seeds h."""
        if self.is_moore:
            state = MooreState('s0', output=self._unwrap_output(self.sul.step(None)))
        else:
            state = MealyState('s0')
        for i in self.input_alphabet:
            output = self.step_wrapper(i)
            state.transitions[i] = state
            if not self.is_moore:
                state.output_fun[i] = output

        machine_class = MooreMachine if self.is_moore else MealyMachine
        mm = machine_class(state, [state])
        mm.current_state = state
        return mm

    def _reset_h_nd_index(self):
        self._hs_cont_starts.clear()
        self._hs_cont_set.clear()
        self._pair_progress.clear()
        self._h_nd_scan_pos = len(self.global_trace)
        self._next_occ_min_start = self._h_nd_scan_pos

    def _reset_state_data(self, reset_h_index=False):
        """Discard the identified states. The h-ND index depends only on h, not on W,
        so W-driven resets keep it (preserving comparison progress and minable data)."""
        self.state_map.clear()
        self.h_response_map.clear()
        self._conjecture_probe_seen.clear()
        if reset_h_index:
            self._reset_h_nd_index()

    def _all_states(self):
        """All known states, complete and partial, without duplicates."""
        seen = set()
        states = []
        for state in list(self.state_map.values()) + list(self.h_response_map.values()):
            if id(state) not in seen:
                seen.add(id(state))
                states.append(state)
        return states

    def check_h_ND_consistency(self):
        """Detect non-determinism of h: two same-response h occurrences whose
        continuations agree on inputs but differ in outputs. On detection h is
        extended with the diverging input sequence and all state data is reset.

        Incremental: the trace is scanned once, and registered continuation pairs
        remember how far they have been compared. Self-overlapping h occurrences
        (e.g. h = i^k inside a longer run of i) are skipped, as their continuations
        share long input prefixes and would grow the pair index quadratically."""
        h = self.homing_sequence
        h_len = len(h)
        if h_len == 0:
            return True

        trace = self.global_trace
        trace_len = len(trace)

        # scan the newly added part of the trace for h occurrences; a new
        # continuation is eagerly paired with all same-response continuations
        scan_start = max(0, self._h_nd_scan_pos - h_len + 1)
        for i in range(scan_start, trace_len - h_len + 1):
            for j in range(h_len):
                if trace[i + j][0] != h[j]:
                    break
            else:
                new_cont = i + h_len
                if new_cont in self._hs_cont_set or i < self._next_occ_min_start:
                    continue
                h_response = tuple(trace[i + j][1] for j in range(h_len))
                for existing in self._hs_cont_starts[h_response]:
                    pair = (new_cont, existing) if new_cont < existing else (existing, new_cont)
                    self._pair_progress[pair] = 0
                self._hs_cont_starts[h_response].append(new_cont)
                self._hs_cont_set.add(new_cont)
                self._next_occ_min_start = new_cont
        self._h_nd_scan_pos = trace_len

        # advance every active pair; pairs whose inputs diverged are deleted
        to_delete = []
        for (p1, p2), already in self._pair_progress.items():
            compare_len = trace_len - p2  # p1 < p2, so p2's continuation is the shorter one
            for k in range(already, compare_len):
                inp1, out1 = trace[p1 + k]
                inp2, out2 = trace[p2 + k]
                if inp1 != inp2:
                    to_delete.append((p1, p2))
                    break
                if out1 != out2:
                    self.homing_sequence += tuple(inp for inp, _ in trace[p1:p1 + k + 1])
                    self.interrupt = True
                    self._reset_state_data(reset_h_index=True)
                    self.add_h_to_W()
                    return False
            else:
                if compare_len > already:
                    self._pair_progress[(p1, p2)] = compare_len

        for pair in to_delete:
            del self._pair_progress[pair]

        return True

    def find_counterexample(self, hypothesis):
        """Random-walk equivalence check (no resets). Returns the executed inputs up
        to and including the first output mismatch, or None."""
        if self.reset_testing_counter:
            current_test_steps = self.num_testing_steps
        else:
            current_test_steps = max(self.num_testing_steps - self.total_testing_steps, 0)

        cex = []
        for _ in range(current_test_steps):
            self.total_testing_steps += 1
            random_input = choice(self.input_alphabet)
            cex.append(random_input)

            if self.step_wrapper(random_input) != hypothesis.step(random_input):
                return cex

        return None

    def is_complete(self):
        """True if every identified state has all transitions leading to identified states."""
        if not self.state_map:
            return False
        return all(
            i in state.transitions and state.transitions[i].hs in self.state_map
            for state in self.state_map.values()
            for i in self.input_alphabet
        )

    def _first_missing_transition_query(self, state):
        """First (input, w) pair whose response is not yet known for this state."""
        for i in self.input_alphabet:
            learned = state.learned_w_per_input[i]
            if len(learned) != len(self.W):
                for w in self.W:
                    if w not in learned:
                        return i, w
        return None

    def _reachable_state_paths(self, start_state):
        """(state, shortest input path) pairs for all states reachable in the conjecture."""
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
                queue.append((next_state, path + (i,)))

        return reachable

    def find_reachable_incomplete_transition(self, start_state):
        """Closest reachable state with an unknown (input, w) response, as
        (path, state, input, w), or None if everything reachable is complete."""
        for state, path in self._reachable_state_paths(start_state):
            missing = self._first_missing_transition_query(state)
            if missing is not None:
                return path, state, missing[0], missing[1]
        return None

    def _simulate_from_state(self, state, sequence):
        """Outputs and end state of running sequence through the conjecture, or
        (None, ...) if the required data is not known yet."""
        if not sequence and self.is_moore:
            state_out = state.state_w_values.get(())
            if state_out is None:
                return None, state
            return (state_out,), state
        outputs = []
        for i in sequence:
            if i not in state.output_fun or i not in state.transitions:
                return None, None
            outputs.append(state.output_fun[i])
            state = state.transitions[i]
        return tuple(outputs), state

    def _empty_suffix_response_after_h(self, h_response):
        """(Moore) The response to the empty suffix is the last output of h,
        so it comes for free with every homing."""
        if not self.is_moore or self.homing_sequence == ():
            return None
        return (h_response[-1],)

    def _mine_h_w_response(self, hs_response, w):
        """Recover the response to w after an h occurrence with hs_response from
        already-observed trace data, avoiding a fresh query."""
        if not w:
            return None
        trace = self.global_trace
        w_len = len(w)
        for cont in self._hs_cont_starts.get(hs_response, ()):
            if cont + w_len > len(trace):
                continue
            for k in range(w_len):
                if trace[cont + k][0] != w[k]:
                    break
            else:
                return tuple(trace[cont + k][1] for k in range(w_len))
        return None

    def _apply_conjecture_probe(self, current_state, path, w, kind):
        """Execute path + h + w on the SUL to expose a suspected inconsistency.
        `kind` distinguishes the check that requested the probe, so each probe runs
        at most once. Returns True if anything was executed."""
        key = (kind, current_state.hs, path, w, self.homing_sequence)
        if key in self._conjecture_probe_seen:
            return False
        self._conjecture_probe_seen.add(key)

        alpha_result = self.execute_conjecture_path(current_state, path)
        if self.interrupt or alpha_result is None:
            return True

        self.execute_homing_sequence()
        if self.interrupt:
            return True

        self.execute_sequence(w)
        self.check_h_ND_consistency()
        return True

    def check_conjecture_inconsistencies(self, current_state):
        """Look for internal inconsistencies of the conjecture: a state whose
        predicted responses after h disagree with the state mapped to that
        h-response, or two states that h cannot separate but W can. A probe is
        executed to expose the first inconsistency found; returns True if so."""
        states_after_h = []
        for state, path in self._reachable_state_paths(current_state):
            h_response, state_after_h = self._simulate_from_state(state, self.homing_sequence)
            if h_response is not None and state_after_h is not None:
                states_after_h.append((h_response, state_after_h, path))

        for h_response, state_after_h, path in states_after_h:
            mapped_state = self.h_response_map.get(h_response)
            if mapped_state is None or len(mapped_state.state_w_values) != len(self.W):
                continue

            for w in self.W:
                actual, _ = self._simulate_from_state(state_after_h, w)
                if actual is not None and mapped_state.state_w_values.get(w) != actual:
                    return self._apply_conjecture_probe(current_state, path, w, 'H')

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

    def _state_for_w_values(self, w_values):
        """State matching these W responses; checked against identified states first,
        then against partially identified ones. Created and registered if not found."""
        profile = self._w_profile(w_values)
        matched = self.state_map.get(profile)
        if matched is None:
            for candidate in self.h_response_map.values():
                if candidate.state_w_values == w_values:
                    matched = candidate
                    break
        if matched is not None:
            return matched

        state = ModelState(profile)
        state.state_w_values = dict(w_values)
        self.state_map[profile] = state
        return state

    def _merge_states(self, canonical, duplicate):
        """Fold everything learned about duplicate into canonical and redirect all
        references to it."""
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
        """Re-key a state identified by its h-response to its full W-profile,
        merging it with an existing state if the profile is already known."""
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

    def update_model_transition(self, state, i):
        """Set state's i-transition once the responses to every w in W are known."""
        learned = state.learned_w_per_input[i]
        if len(learned) != len(self.W):
            return
        w_for_input = {w: state.transition_w_values[(i, w)] for w in learned}
        state.transitions[i] = self._state_for_w_values(w_for_input)

    def _partition_signature(self, state, block_of):
        """Refinement signature: state output (Moore) or transition outputs (Mealy),
        plus the current block of each successor."""
        signature = [state.state_w_values.get(())] if self.is_moore else []
        for i in self.input_alphabet:
            successor = state.transitions.get(i)
            block = block_of.get(successor.hs) if successor is not None else None
            if self.is_moore:
                signature.append((i, block))
            else:
                signature.append((i, state.output_fun.get(i), block))
        return tuple(signature)

    def _state_partitions(self):
        """Group equivalent identified states into blocks via partition refinement.
        Returns hs -> block index."""
        states = [s for s in self.state_map.values() if len(s.state_w_values) == len(self.W)]

        block_of = {}
        while True:
            signatures = {}
            next_block_of = {}
            for state in states:
                signature = self._partition_signature(state, block_of)
                next_block_of[state.hs] = signatures.setdefault(signature, len(signatures))
            if next_block_of == block_of:
                return block_of
            block_of = next_block_of

    def create_model(self, current_hs):
        """Build a Moore/Mealy machine from the state blocks, starting (and keeping
        only states reachable) from the state identified by current_hs."""
        block_of = self._state_partitions()

        # one automaton state per block, built from the block's first member
        block_states = {}
        for hs, block in block_of.items():
            if block in block_states:
                continue
            model_state = self.state_map[hs]
            state_id = f's{len(block_states)}'
            if self.is_moore:
                raw_out = model_state.state_w_values.get(())
                state_output = raw_out[0] if isinstance(raw_out, tuple) and len(raw_out) == 1 else raw_out
                aut_state = MooreState(state_id, output=state_output)
            else:
                aut_state = MealyState(state_id)
            aut_state.prefix = hs
            block_states[block] = aut_state

        self._aut_state_by_hs = {hs: block_states[block] for hs, block in block_of.items()}

        for hs, block in block_of.items():
            if block_states[block].prefix != hs:
                continue  # transitions are taken from the block representative only
            state = self.state_map[hs]
            automata_state = block_states[block]
            for i, successor in state.transitions.items():
                target_block = block_of.get(successor.hs)
                if target_block is not None:
                    automata_state.transitions[i] = block_states[target_block]
                    if not self.is_moore:
                        automata_state.output_fun[i] = state.output_fun[i]

        start_state = self._aut_state_by_hs.get(current_hs)
        if start_state is None:
            start_state = next(iter(block_states.values()))

        # keep only states reachable from the start state
        reachable_states = []
        queue = deque([start_state])
        seen = set()
        while queue:
            state = queue.popleft()
            if state in seen:
                continue
            seen.add(state)
            reachable_states.append(state)
            queue.extend(state.transitions.values())

        # close still-unknown transitions with self-loops
        for s in reachable_states:
            for i in self.input_alphabet:
                if i not in s.transitions:
                    s.transitions[i] = s
                    if not self.is_moore:
                        s.output_fun.setdefault(i, None)

        machine_class = MooreMachine if self.is_moore else MealyMachine
        mm = machine_class(start_state, reachable_states)
        mm.current_state = start_state
        return mm

    def _track_through_conjecture(self, target, x, w, w_response):
        """Follow the conjecture from target through x and w. Returns the state the
        SUL is in after the probe, or None if any transition along the way is unknown
        or a recorded output disagrees with the observed w_response."""
        state = target.transitions.get(x)
        if state is None:
            return None
        for idx, inp in enumerate(w):
            next_state = state.transitions.get(inp)
            if next_state is None or state.output_fun.get(inp) != w_response[idx]:
                return None
            state = next_state
        return state

    def create_hypothesis(self):
        """Main learning loop: localize via h, identify the current state's W
        responses, then learn outgoing transitions of reachable states until the
        conjecture is complete and consistent."""
        # When the SUL's current state is known (nothing was executed since the last
        # localization, or the conjecture fully predicts the probe just executed),
        # the next iteration skips the homing sequence entirely.
        tracked_state = None
        while True:
            if tracked_state is not None:
                current_state = tracked_state
                hs_response = current_state.hs  # profile stands in for the h-response in cache keys
                tracked_state = None
            else:
                hs_response = self.execute_homing_sequence()

                if self.interrupt:
                    self.interrupt = False
                    continue

                current_state = self.h_response_map.get(hs_response)
                if current_state is None:
                    current_state = ModelState(hs_response)
                    self.h_response_map[hs_response] = current_state

            if len(current_state.state_w_values) != len(self.W):
                # identify the current state: resolve as many w responses as possible
                # without moving the SUL (cache, Moore shortcut, trace mining) and
                # execute at most one w query per homing
                executed_query = False
                for w in self.W:
                    if w in current_state.state_w_values:
                        continue
                    dict_key = (self.homing_sequence, hs_response, w)
                    w_response = self.h_w_dictionary.get(dict_key)
                    if w_response is None and w == ():
                        w_response = self._empty_suffix_response_after_h(hs_response)
                    if w_response is None:
                        w_response = self._mine_h_w_response(hs_response, w)
                    if w_response is None:
                        w_response = self.execute_sequence(w)
                        executed_query = True
                        if self.interrupt:
                            break
                    self.h_w_dictionary[dict_key] = w_response
                    current_state.state_w_values[w] = w_response
                    if executed_query:
                        break
                if not self.interrupt and len(current_state.state_w_values) == len(self.W):
                    current_state = self._complete_h_response_state(hs_response, current_state)
                    if not executed_query:
                        # completed without moving the SUL: still located at this state
                        tracked_state = current_state
            else:
                reachable_incomplete = self.find_reachable_incomplete_transition(current_state)
                if reachable_incomplete is None:
                    if self.check_conjecture_inconsistencies(current_state):
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
                    if self.is_moore and w == ():
                        w_response = (output,)
                    else:
                        w_response = self.execute_sequence(w)

                    if not self.interrupt:
                        self.dictionary[dict_key] = (alpha_response, output, w_response)
                else:
                    _, output, w_response = cached
                    # answered from cache, the SUL did not move
                    tracked_state = current_state

                if self.interrupt:
                    self.interrupt = False
                    tracked_state = None
                    continue

                target.transition_w_values[(x, w)] = w_response
                target.learned_w_per_input[x].add(w)
                target.output_fun[x] = output

                self.update_model_transition(target, x)

                if tracked_state is None:
                    tracked_state = self._track_through_conjecture(target, x, w, w_response)

        return self.create_model(current_state.hs)

    def main_loop(self, print_level=2):
        """Outer loop: bootstrap h, alternate hypothesis construction with random-walk
        equivalence checks, refine W from counterexamples, and assemble the result."""
        start_time = time.time()
        eq_query_time = 0

        if not self._user_provided_h:
            # h starts as the last input of the daisy hypothesis' first counterexample
            initial_model = self.create_daisy_hypothesis()

            eq_start = time.time()
            counter_example = self.find_counterexample(initial_model)
            eq_query_time += time.time() - eq_start

            last_cex_input = (counter_example[-1],)
            self.homing_sequence = last_cex_input
            if self.is_moore:
                self.add_to_W(())
            self.add_to_W(last_cex_input)

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

            # extend W with the shortest new suffix of the counterexample
            added_suffix = None
            for s in sorted((tuple(s) for s in all_suffixes(counter_example)), key=len):
                if self.add_to_W(s):
                    added_suffix = s
                    break

            if added_suffix is None:
                if not self.add_to_W(self.homing_sequence + tuple(counter_example)):
                    break

            self.add_h_to_W()
            self._reset_state_data()
            self.interrupt = False

        # finalize initial reset cleanup
        self.sul.post()

        queries_before_initial_state = self.sul.num_queries

        if self.query_for_initial_state:
            # find the model state matching the SUL's reset behavior, by W responses
            # first and by h-response as a fallback
            initial_w_values = {w: tuple(self.sul.query(w)) for w in self.W}
            initial_model_state = next(
                (ms for ms in self.state_map.values() if ms.state_w_values == initial_w_values), None)
            if initial_model_state is None:
                initial_hs_response = tuple(self.sul.query(self.homing_sequence))
                initial_model_state = next(
                    (ms for ms in self.state_map.values()
                     if self._simulate_from_state(ms, self.homing_sequence)[0] == initial_hs_response), None)
            if initial_model_state is not None:
                candidate = self._aut_state_by_hs.get(initial_model_state.hs)
                if candidate is not None and candidate in hypothesis.states:
                    hypothesis.initial_state = candidate

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


def run_hW(alphabet: list, sul, automaton_type='mealy', num_testing_steps=200, reset_testing_counter=True,
           query_for_initial_state=True, H=None, W=None, return_data=False, print_level=2):
    """
    Executes the hW resetless learning algorithm.
    Algorithm description can be found in "hW-inference: A heuristic approach to retrieve models through
    black box testing" by Groz et al.

    The implementation does not strictly follow all aspects of the described algorithm, but relies on in for the most
    part.

    Args:

        alphabet: input alphabet

        sul: system under learning

        automaton_type: type of automaton to be learned. Either 'mealy', 'moore', or 'dfa'. For 'moore' and 'dfa'
            the algorithm treats outputs as state properties (Moore semantics). 'dfa' additionally casts the final
            Moore machine to a Dfa, treating True/False outputs as accepting/rejecting states. (Default value = 'mealy')

        num_testing_steps: number of random steps used per equivalence check (Default value = 200)

        reset_testing_counter: reset the testing step counter after each counterexample (Default value = True)

        query_for_initial_state: if True, query the SUL to identify the true initial state (Default value = True)

        H: optional user-provided homing sequence. If supplied, hW starts with this sequence instead of deriving an
            initial one from the daisy hypothesis counterexample. (Default value = None)

        W: optional user-provided characterization set. If supplied, hW starts with these suffixes instead of an empty
            characterization set. For Moore machines and DFAs, the empty suffix is added automatically.
            (Default value = None)

        return_data: if True, return a (hypothesis, info) tuple instead of just the hypothesis
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

    Returns:

        learned automaton of type automaton_type (or (automaton, info dict) if return_data is True)
    """
    assert print_level in [0, 1, 2, 3]
    assert automaton_type in ('mealy', 'moore', 'dfa')

    # DFAs are learned as Moore machines and cast to a Dfa afterwards
    hw = hW(
        input_al=alphabet,
        sul=sul,
        automaton_type='moore' if automaton_type == 'dfa' else automaton_type,
        num_testing_steps=num_testing_steps,
        reset_testing_counter=reset_testing_counter,
        query_for_initial_state=query_for_initial_state,
        H=H,
        W=W,
    )

    hypothesis, info = hw.main_loop(print_level=print_level)

    if automaton_type == 'dfa':
        dfa_states = []
        state_map = {}
        for moore_s in hypothesis.states:
            dfa_s = DfaState(moore_s.state_id, is_accepting=bool(moore_s.output))
            dfa_s.prefix = getattr(moore_s, 'prefix', None)
            state_map[moore_s] = dfa_s
            dfa_states.append(dfa_s)
        for moore_s in hypothesis.states:
            dfa_s = state_map[moore_s]
            for letter, target in moore_s.transitions.items():
                dfa_s.transitions[letter] = state_map[target]
        hypothesis = Dfa(state_map[hypothesis.initial_state], dfa_states)
        hypothesis.current_state = hypothesis.initial_state

    if return_data:
        return hypothesis, info

    return hypothesis
