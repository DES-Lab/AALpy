class Node(object):
    __slots__ = ['value', 'children']

    def __init__(self, value=None):
        self.value = value
        self.children = {}


class CacheTree:
    """
    Tree in which all membership queries and corresponding outputs/values are stored. Membership queries update the tree
    and while updating, check if determinism is maintained.
    Root node corresponds to the initial state, and from that point on, for every new input/output pair, a new child is
    created where the output is the value of the child, and the input is the transition leading from the parent to the
    child.
    """

    def __init__(self):
        self.root_node = Node()
        self.curr_node = None
        self.inputs = ()
        self.outputs = ()

    def reset(self):
        self.curr_node = self.root_node
        self.inputs = ()
        self.outputs = ()

    def step_in_cache(self, inp, out):
        """
        Preform a step in the cache. If output exist for the current state, and is not the same as `out`, throw
        the non-determinism violation error and abort learning.
        Args:

            inp: input
            out: output

        """
        self.inputs += (inp,)
        self.outputs += (out,)
        if inp is None:
            self.root_node.value = out
            return

        if inp not in self.curr_node.children.keys():
            node = Node(out)
            self.curr_node.children[inp] = node
        else:
            node = self.curr_node.children[inp]
            if node.value != out:
                expected_seq = self.outputs[:-1]
                expected_seq += (node.value,)
                msg = f'Non-determinism detected.\n' \
                      f'Error inserting: {self.inputs}\n' \
                      f'Conflict detected: {node.value} vs {out}\n' \
                      f'Expected Output: {expected_seq}\n' \
                      f'Received output: {self.outputs}'
                raise SystemExit(msg)
        self.curr_node = node

    def in_cache(self, input_seq: tuple):
        """
        Check if the result of the membership query for input_seq is cached is in the tree. If it is, return the
        corresponding output sequence.

        Args:

            input_seq: corresponds to the membership query

        Returns:

            outputs associated with inputs if it is in the query, None otherwise

        """
        curr_node = self.root_node

        output_seq = ()
        for letter in input_seq:
            if letter in curr_node.children.keys():
                curr_node = curr_node.children[letter]
                output_seq += (curr_node.value,)
            else:
                return None

        return output_seq

    def add_to_cache(self, input_sequence, output_sequence):
        """
        Add input-output sequence to cache
        """
        self.reset()
        for i, o in zip(input_sequence, output_sequence):
            self.step_in_cache(i, o)


class CacheDict:
    """
    Dictionary in which all membership queries and corresponding outputs/values are stored. Membership queries update
    the tree and while updating, check if determinism is maintained.
    Root node corresponds to the initial state, and from that point on, for every new input/output pair, a new child is
    created where the output is the value of the child, and the input is the transition leading from the parent to the
    child.
    """

    def __init__(self):
        self.cache_dict = dict()
        self.inputs = ()

    def reset(self):
        self.inputs = ()
        pass

    def step_in_cache(self, inp, out):
        """
        Preform a step in the cache. If output exist for the current state, and is not the same as `out`, throw
        the non-determinism violation error and abort learning.
        Args:

            inp: input
            out: output

        """

        if inp is None:
            return self.cache_dict[()]

        self.inputs += (inp,)

        if self.inputs not in self.cache_dict.keys():
            self.cache_dict[self.inputs] = out
        else:
            cache_output = self.cache_dict[self.inputs]
            if cache_output != out:
                expected_seq = self.get_output_sequence(self.inputs)
                received_seq = expected_seq[:-1] + (out,)
                msg = f'Non-determinism detected.\n' \
                      f'Error inserting: {self.inputs}\n' \
                      f'Conflict detected: {cache_output} vs {out}\n' \
                      f'Expected Output: {expected_seq}\n' \
                      f'Received output: {received_seq}'
                raise SystemExit(msg)

    def in_cache(self, input_seq: tuple):
        """
        Check if the result of the membership query for input_seq is cached is in the tree. If it is, return the
        corresponding output sequence.

        Args:

            input_seq: corresponds to the membership query

        Returns:

            outputs associated with inputs if it is in the query, None otherwise

        """
        if input_seq in self.cache_dict.keys():
            return self.get_output_sequence(input_seq)
        return None

    def add_to_cache(self, input_sequence, output_sequence):
        """
        Add input-output sequence to cache
        """
        for i in range(1, len(input_sequence) + 1):
            self.cache_dict[input_sequence[:i]] = output_sequence[i-1]

    def get_output_sequence(self, input_seq):
        return tuple(self.cache_dict[input_seq[:i]] for i in range(1, len(input_seq) + 1))
