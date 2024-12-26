import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL

class DiffFirstEqOracle(Oracle):
    """
    Equivalence oracle that first explores the 'difference' of the current hypothesis to the previous
    one. The intuition behind this is that, if a new error is to occur, then it is likely to occur in the
    'new' part of the hypothesis, that has not been examined before.
    """
    raise NotImplementedError

