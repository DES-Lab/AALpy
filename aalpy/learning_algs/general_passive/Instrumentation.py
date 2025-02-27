import time
from typing import Dict, Optional

from aalpy.learning_algs.general_passive.GeneralizedStateMerging import Instrumentation, Partitioning, \
    GeneralizedStateMerging
from aalpy.learning_algs.general_passive.GsmNode import GsmNode


class ProgressReport(Instrumentation):
    def __init__(self, lvl):
        super().__init__()
        self.lvl = lvl
        if lvl < 1:
            return
        self.gsm: Optional[GeneralizedStateMerging] = None
        self.log = []
        self.pta_size = None
        self.nr_merged_states_total = 0
        self.nr_merged_states = 0
        self.nr_red_states = 0

        self.previous_time = None

    def reset(self, gsm: GeneralizedStateMerging):
        self.gsm = gsm
        self.log = []
        self.pta_size = None
        self.nr_merged_states_total = 0
        self.nr_merged_states = 0
        self.nr_red_states = 0

        self.previous_time = time.time()

    def pta_construction_done(self, root):
        print(f'PTA Construction Time: {round(time.time() - self.previous_time, 2)}')
        if 1 < self.lvl:
            states = root.get_all_nodes()
            leafs = [state for state in states if len(state.transitions.keys()) == 0]
            depth = [state.get_prefix_length() for state in leafs]
            self.pta_size = len(states)
            print(f'PTA has {len(states)} states leading to {len(leafs)} leafs')
            print(f'min / avg / max depth : {min(depth)} / {sum(depth) / len(depth)} / {max(depth)}')
        self.previous_time = time.time()

    def print_status(self):
        reset_char = "\33[2K\r"
        print_str = reset_char + f'Current automaton size: {self.nr_red_states}'
        if 1 < self.lvl and not self.gsm.compatibility_on_futures:
            print_str += f' Merged: {self.nr_merged_states_total} Remaining: {self.pta_size - self.nr_red_states - self.nr_merged_states_total}'
        print(print_str, end="")

    def log_promote(self, node: GsmNode):
        self.log.append(["promote", (node.get_prefix(),)])
        self.nr_red_states += 1
        self.print_status()

    def log_merge(self, part: Partitioning):
        self.log.append(["merge", (part.red.get_prefix(), part.blue.get_prefix())])
        self.nr_merged_states_total += len(part.full_mapping) - len(part.red_mapping)
        self.nr_merged_states += 1
        self.print_status()

    def learning_done(self, root, red_states):
        print(f'\nLearning Time: {round(time.time() - self.previous_time, 2)}')
        print(f'Learned {len(red_states)} state automaton via {self.nr_merged_states} merges.')
        if 2 < self.lvl:
            root.visualize("model", self.gsm.output_behavior)


class MergeViolationDebugger(Instrumentation):
    def __init__(self, ground_truth: GsmNode):
        super().__init__()
        self.root = ground_truth
        self.map: Dict[GsmNode, GsmNode] = dict()
        self.log = []
        self.gsm: Optional[GeneralizedStateMerging] = None

    def reset(self, gsm: GeneralizedStateMerging):
        self.gsm = gsm
        self.map = dict()
        self.log = []

    def log_promote(self, new_red: GsmNode):
        new_red_prefix = new_red.get_prefix()
        node = self.root.get_by_prefix(new_red_prefix)
        old_red = self.map.get(node)
        if old_red is None:
            self.map[node] = new_red
            self.log.append(("promote", new_red_prefix))
        elif old_red is not new_red:
            print(f"Erroneous promotion detected:")
            print(f"  Ground truth: {node.get_prefix()}")
            print(f"  Representative (old): {old_red.get_prefix()}")
            print(f"  Representative (new): {new_red_prefix}")
            self.log.append(("wrong promote", new_red_prefix))

    def log_merge(self, part: Partitioning):
        red_prefix = part.red.get_prefix()
        blue_prefix = part.blue.get_prefix()
        red_node = self.root.get_by_prefix(red_prefix)
        blue_node = self.root.get_by_prefix(blue_prefix)
        if red_node is None or blue_node is None:
            self.log.append(("broken merge", red_prefix, blue_prefix))
        elif red_node is blue_node:
            self.log.append(("merge", red_prefix, blue_prefix))
        else:
            print(f"Erroneous merge detected:")
            print(f"  PTA red: {red_prefix}")
            print(f"  PTA blue: {blue_prefix}")
            print(f"  real red: {red_node.get_prefix()}")
            print(f"  real blue: {blue_node.get_prefix()}")
            self.log.append(("wrong merge", red_prefix, blue_prefix))
