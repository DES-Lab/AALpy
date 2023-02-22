import queue
import time
from typing import Tuple

from aalpy.utils import save_automaton_to_file
from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import to_automaton, RpniNode, StateMerging, \
  AutomatonType, createPTA


class GeneralizedStateMerging:
  def __init__(self, data, automaton_type : AutomatonType, print_info=True):
    self.data = data
    automaton_type = AutomatonType[automaton_type] if isinstance(automaton_type, str) else automaton_type
    self.automaton_type = automaton_type
    self.print_info = print_info

    pta_construction_start = time.time()
    self.merger = StateMerging(data, automaton_type)
    self.log = []

    if self.print_info:
      print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')

  def run(self):
    start_time = time.time()

    # sorted list of states already considered
    red_states = [self.merger.root]
    # used to get the minimal non-red state
    blue_states = list(red_states[0].children.values())

    while blue_states:
      blue_state = min(list(blue_states), key=lambda x: len(x.prefix))

      partition = None
      for red_state in red_states:
        partition = self._partition_from_merge(red_state, blue_state)
        if partition is not None:
          break

      if partition is None:
        self.log.append(["promote", (blue_state.prefix,)])
        red_states.append(blue_state)
      else:
        self.log.append(["merge", (red_state.prefix, blue_state.prefix)])

        # use the partition for merging
        # TODO isolate and move to RpniNode?
        for node in partition.keys():
          block = partition[node]
          #assert RpniNode.compatible(node, block)
          node.output = block.output
          node.children = block.children

        node = self.merger.root.get_child_by_prefix(blue_state.prefix[:-1])
        node.children[blue_state.prefix[-1]] = red_state
        #self.merge(red_state, blue_state)

      blue_states.clear()
      for r in red_states:
        for c in r.children.values():
          if c not in red_states:
            blue_states.append(c)

    if self.print_info:
      print(f'\nGSM Learning Time: {round(time.time() - start_time, 2)}')
      print(f'GSM Learned {len(red_states)} state automaton.')

    return to_automaton(red_states, self.automaton_type)

  def merge(self, red_state, blue_state):
    if not self.merger.merge(red_state, blue_state):
      print(f"error on command: {self.log[-1]}")
      v1 = StateMerging.replay_log_on_pta(self.data, self.log[:-1], self.automaton_type)
      v2 = StateMerging.replay_log_on_pta(self.data, self.log, self.automaton_type)
      save_automaton_to_file(v1, "pre", "pdf")
      save_automaton_to_file(v2, "post", "pdf")
      raise AssertionError(f"error on command: {self.log[-1]}")

  def _compatible_states_future(self, red, blue):
    """
    Compatibility check based on futures
    """

    if self.automaton_type == AutomatonType.mealy:
      raise NotImplementedError()

    # TODO move to RpniNode and generalize to non-tree values blue_node
    # this is done by tracking which (red,blue) pairs have been visited

    overwrites = []
    def revert_overrides():
      for overwrite in overwrites:
        overwrite.output = None

    q : queue.Queue[Tuple[RpniNode,RpniNode]] = queue.Queue()
    q.put((red, blue))
    while not q.empty():
      red, blue = q.get()
      if not RpniNode.compatible_outputs(red, blue):
        revert_overrides()
        return False
      if red.output is None and blue.output is not None:
        red.output = blue.output
        overwrites.append(red)
      for symbol in blue.children.keys():
        if symbol in red.children.keys():
          q.put((red.children[symbol],blue.children[symbol]))
    revert_overrides()
    return True

  def _partition_from_merge(self, red : RpniNode, blue : RpniNode):
    """
    Compatibility check based on partitions
    """

    # TODO use partitions for determining the next state to consider
    # Outline:
    # start with sorted list of nodes
    # throw out nodes whenever they are merged with a red node

    # TODO add possibility to calculate (custom) scores of the merge candidates
    # TODO add option to loosen determinism to output determinism for use in alergia
    # TODO make associated data (such as observation frequency) availabel to cost function

    # TODO move to RpniNode and generalize to non-tree values for blue?
    # maybe create partition first and check compatibility afterwards?

    partitions = dict()
    q = queue.Queue()
    q.put((red, blue))

    while not q.empty():
      red, blue = q.get()

      def get_partition(node : RpniNode):
        if node not in partitions:
          p = node.shallow_copy()
          partitions[node] = p
        else:
          p = partitions[node]
        return p

      partition = get_partition(red)

      if not RpniNode.compatible_outputs(partition, blue):
        return None
      if self.automaton_type == AutomatonType.moore and partition.output is None:
        partition.output = blue.output
      if self.automaton_type == AutomatonType.mealy:
        for key in filter(lambda k : k not in red.output or red.output[k] is None, blue.output):
          partition.output[key] = blue.output[key]

      partitions[blue] = partition

      for symbol, blue_child in blue.children.items():
        if symbol in partition.children.keys():
          q.put((partition.children[symbol], blue_child))
        else :
          # blue_child is blue after merging if there is a red state in the partition
          partition.children[symbol] = blue_child
    return partitions

