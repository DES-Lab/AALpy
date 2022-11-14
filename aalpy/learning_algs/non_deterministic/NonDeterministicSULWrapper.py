from aalpy.base import SUL
from aalpy.learning_algs.non_deterministic.TraceTree import TraceTree


class NonDeterministicSULWrapper(SUL):
    """
    Wrapper for non-deterministic SUL. After every step, input/output pair is added to the tree containing all traces.
    """

    def __init__(self, sul: SUL):
        super().__init__()
        self.sul = sul
        self.cache = TraceTree()

    def pre(self):
        self.cache.reset()
        self.sul.pre()

    def post(self):
        self.sul.post()

    def step(self, letter):
        out = self.sul.step(letter)
        self.cache.add_to_tree(letter, out)
        return out
