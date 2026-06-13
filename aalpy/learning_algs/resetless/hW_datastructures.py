from collections import defaultdict


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


class HomingSequenceIndex:
    """
    Incremental index over the global trace used to detect non-determinism of the
    homing sequence h: two same-response h occurrences whose continuations agree on
    inputs but differ in outputs.

    The trace is scanned once, and registered continuation pairs remember how far
    they have been compared. Self-overlapping incidental h occurrences (e.g.
    h = i^k inside a longer run of i) are skipped, as their continuations share long
    input prefixes and would grow the pair index quadratically. Deliberate homing
    executions are exempt from that skip (forced_cont_start), otherwise their
    non-determinism could go undetected.
    """

    def __init__(self):
        self._hs_cont_starts = defaultdict(list)   # h-response -> [continuation start positions]
        self._hs_cont_set = set()                  # all registered continuation starts (O(1) membership)
        self._pair_progress = {}                   # (p1, p2) -> compared continuation length so far
        self._scan_pos = 0                         # how far the trace has been scanned
        self._next_occ_min_start = 0               # earliest start of the next registered h occurrence

    def reset(self, trace_len):
        """
        Clear the index. The old trace was produced under the previous h and can
        contain many incidental occurrences of the extended h; rescanning it makes
        h grow and forces avoidable relearning.
        """
        self._hs_cont_starts.clear()
        self._hs_cont_set.clear()
        self._pair_progress.clear()
        self._scan_pos = trace_len
        self._next_occ_min_start = trace_len

    def continuation_starts(self, hs_response):
        """Continuation start positions recorded for the given h-response."""
        return self._hs_cont_starts.get(hs_response, ())

    def scan(self, trace, h, forced_cont_start=None):
        """
        Register h occurrences in the newly added part of the trace, advance every
        active continuation pair, and report non-determinism of h. Returns the
        diverging input sequence to extend h with (the shortest witness among all
        pairs that diverged in this call), or None if h is still consistent.
        """
        h_len = len(h)
        if h_len == 0:
            return None

        trace_len = len(trace)

        # scan the newly added part of the trace for h occurrences; a new
        # continuation is eagerly paired with all same-response continuations
        scan_start = max(0, self._scan_pos - h_len + 1)
        for i in range(scan_start, trace_len - h_len + 1):
            for j in range(h_len):
                if trace[i + j][0] != h[j]:
                    break
            else:
                new_cont = i + h_len
                if new_cont in self._hs_cont_set:
                    continue
                if i < self._next_occ_min_start and new_cont != forced_cont_start:
                    continue
                h_response = tuple(trace[i + j][1] for j in range(h_len))
                for existing in self._hs_cont_starts[h_response]:
                    pair = (new_cont, existing) if new_cont < existing else (existing, new_cont)
                    self._pair_progress[pair] = 0
                self._hs_cont_starts[h_response].append(new_cont)
                self._hs_cont_set.add(new_cont)
                self._next_occ_min_start = new_cont
        self._scan_pos = trace_len

        # advance every active pair; pairs whose inputs diverged are deleted; among
        # output divergences found in this call the shortest witness extends h
        to_delete = []
        divergence = None  # (k, p1) of the minimal output divergence
        for (p1, p2), already in self._pair_progress.items():
            compare_len = trace_len - p2  # p1 < p2, so p2's continuation is the shorter one
            for k in range(already, compare_len):
                inp1, out1 = trace[p1 + k]
                inp2, out2 = trace[p2 + k]
                if inp1 != inp2:
                    to_delete.append((p1, p2))
                    break
                if out1 != out2:
                    if divergence is None or k < divergence[0]:
                        divergence = (k, p1)
                    to_delete.append((p1, p2))
                    break
            else:
                if compare_len > already:
                    self._pair_progress[(p1, p2)] = compare_len

        if divergence is not None:
            k, p1 = divergence
            return tuple(inp for inp, _ in trace[p1:p1 + k + 1])

        for pair in to_delete:
            del self._pair_progress[pair]

        return None
