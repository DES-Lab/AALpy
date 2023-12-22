from math import sqrt, log
from typing import Callable, Any

from aalpy.learning_algs.stochastic_passive.helpers import Node

Score = bool | float
ScoreFunction = Callable[[Node, Node, Any, bool], Score]

def hoeffding_compatibility(eps) -> ScoreFunction:
    def similar(a: Node, b: Node, _: Any, compare_original):
        for in_sym in filter(lambda x : x in a.transitions.keys(), b.transitions.keys()):
            # could create appropriate dict here
            a_trans, b_trans = (x.transitions[in_sym] for x in [a,b])
            if compare_original:
                a_total, b_total = (sum(x.original_count for x in x.values()) for x in (a_trans, b_trans))
            else:
                a_total, b_total = (sum(x.count for x in x.values()) for x in (a_trans, b_trans))
            if a_total == 0 or b_total == 0:
                continue
            threshold = ((sqrt(1 / a_total) + sqrt(1 / b_total)) * sqrt(0.5 * log(2 / eps)))
            for out_sym in set(a_trans.keys()).union(b_trans.keys()):
                if compare_original:
                    ac, bc = (x[out_sym].original_count if out_sym in x else 0 for x in (a_trans, b_trans))
                else:
                    ac, bc = (x[out_sym].count if out_sym in x else 0 for x in (a_trans, b_trans))
                if abs(ac / a_total - bc / b_total) > threshold:
                    return False
        return True
    return similar

def non_det_compatibility(eps) -> ScoreFunction:
    print("Warning: using experimental compatibility criterion for nondeterministic automata")
    def similar(a: Node, b: Node, _: Any, compare_original : bool):
        if compare_original:
            raise NotImplementedError()
        for in_sym in filter(lambda x : x in a.transitions.keys(), b.transitions.keys()):
            a_trans, b_trans = (x.transitions[in_sym] for x in [a,b])
            a_total, b_total = (sum(x.count for x in x.values()) for x in (a_trans, b_trans))
            if a_total < eps or b_total < eps:
                continue
            if set(a_trans.keys()) != set(b_trans.keys()):
                return False
        return True
    return similar