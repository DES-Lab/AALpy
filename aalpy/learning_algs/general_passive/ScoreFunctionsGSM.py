from collections import defaultdict
from math import sqrt, log, lgamma
from typing import Callable, Dict, Union, List, Iterable

from aalpy.learning_algs.general_passive.helpers import Node

Score = Union[bool, float]
LocalScoreFunction = Callable[[Node, Node], Score]
GlobalScoreFunction = Callable[[Dict[Node, Node]], Score]
AggregationFunction = Callable[[Iterable[Score]], Score]

class ScoreCalculation:
    def __init__(self, local_score : LocalScoreFunction = None, global_score : GlobalScoreFunction = None):
        # This is a hack that gives an simple implementation where we can easily
        # - determine whether the default is overridden (for optimization)
        # - override behavior in a functional way by providing the functions as arguments (no extra class)
        # - override behavior in a stateful way by implementing a new class
        if not hasattr(self, "local_score"):
            self.local_score : LocalScoreFunction = local_score or self.default_local_score
        if not hasattr(self, "global_score"):
            self.global_score : GlobalScoreFunction = global_score or self.default_global_score

    def reset(self):
        pass

    @staticmethod
    def default_local_score(a, b):
        return True

    @staticmethod
    def default_global_score(part):
        return True

    def has_default_global_score(self):
        return self.global_score is self.default_global_score

    def has_default_local_score(self):
        return self.local_score is self.default_local_score

def hoeffding_compatibility(eps, compare_original = True) -> LocalScoreFunction:
    eps_fact = sqrt(0.5 * log(2 / eps))
    count_name = "original_count" if compare_original else "count"

    def similar(a: Node, b: Node):
        for in_sym in filter(lambda x : x in a.transitions.keys(), b.transitions.keys()):
            # could create appropriate dict here
            a_trans, b_trans = (x.transitions[in_sym] for x in [a,b])
            a_total, b_total = (sum(getattr(x, count_name) for x in trans.values()) for trans in (a_trans, b_trans))
            if a_total == 0 or b_total == 0:
                continue
            threshold = eps_fact * (sqrt(1 / a_total) + sqrt(1 / b_total))
            for out_sym in set(a_trans.keys()).union(b_trans.keys()):
                ac, bc = (getattr(x[out_sym], count_name) if out_sym in x else 0 for x in (a_trans, b_trans))
                if abs(ac / a_total - bc / b_total) > threshold:
                    return False
        return True
    return similar

def non_det_compatibility(allow_subset = False) -> LocalScoreFunction:
    def compat(a : Node, b : Node):
        for in_sym, a_trans in a.transitions.items():
            b_trans = b.transitions.get(in_sym)
            if b_trans is None:
                continue
            c1 = allow_subset and any(out_sym not in a_trans for out_sym in b_trans.keys())
            c2 = not allow_subset and set(a_trans.keys()) != set(b_trans.keys())
            if c1 or c2:
                return False
        return True
    return compat

class NoRareEventNonDetScore(ScoreCalculation):
    def __init__(self, thresh, p_min : Union[dict, float], reject_local_score_only = False, no_global_score = False):
        super().__init__()
        print("Warning: using experimental compatibility criterion for nondeterministic automata")
        self.thresh = log(thresh)
        if isinstance(p_min, float):
            cost = log(1 - p_min)
            self.miss_dict = defaultdict(lambda: cost)
        else:
            self.miss_dict = {k: log(1 - v) for k, v in p_min.items()}
        self.score = 0
        self.reject_local_score_only = reject_local_score_only
        if no_global_score:
            self.global_score = self.default_global_score

    def reset(self):
        self.score = 0

    def local_score(self, a: Node, b: Node):
        score_local = 0
        for in_sym in filter(lambda x: x in a.transitions.keys(), b.transitions.keys()):
            a_trans, b_trans = (x.transitions[in_sym] for x in [a, b])
            a_total, b_total = (sum(x.count for x in x.values()) for x in (a_trans, b_trans))
            for out_sym in set(a_trans.keys()).union(b_trans.keys()):
                a_miss, b_miss = (out_sym not in trans for trans in [a_trans, b_trans])
                if a_miss:
                    score_local += a_total * self.miss_dict[in_sym, out_sym]
                if b_miss:
                    score_local += b_total * self.miss_dict[in_sym, out_sym]

        self.score += score_local
        if self.reject_local_score_only:
            return self.thresh < score_local
        return self.thresh < self.score

    def global_score(self, part: Dict[Node, Node]) -> Score:
        # I don't think that we have to reevaluate on the full partition.
        return self.score

class KTailCondition(ScoreCalculation):
    def __init__(self, k):
        super().__init__()
        self.depth_offset = None
        self.k = k

    def reset(self):
        self.depth_offset = None

    def local_score(self, a : Node, b : Node):
        # assuming b is tree shaped.
        if self.depth_offset is None:
            self.depth_offset = len(b.prefix)
        depth = len(b.prefix) - self.depth_offset
        return self.k <= depth

class ScoreCombinator(ScoreCalculation):
    def __init__(self, scores : List[ScoreCalculation], aggregate_local : AggregationFunction = None, aggregate_global : AggregationFunction = None):
        super().__init__()
        self.scores = scores
        self.aggregate_local = aggregate_local or all
        self.aggregate_global = aggregate_global or (lambda x : list(x))

    def reset(self):
        for score in self.scores:
            score.reset()

    def local_score(self, a : Node, b : Node):
        return self.aggregate_local(score.local_score(a, b) for score in self.scores)

    def global_score(self, part : Dict[Node, Node]):
        return self.aggregate_global(score.global_score(part) for score in self.scores)

def local_to_global_score(local_fun : LocalScoreFunction) -> GlobalScoreFunction:
    """
    Converts a local score function to a global score function.
    This can be used to evaluate a local score function after the partitions are complete.
    """
    def fun(part : Dict[Node, Node]):
        for old_node, new_node in part.items():
            if local_fun(new_node, old_node) is False:
                return False
        return True
    return fun

def differential_info(part : Dict[Node, Node]):
    relevant_nodes_old = list(part.keys())
    relevant_nodes_new = set(part.values())

    partial_llh_old = sum(node.local_log_likelihood_contribution() for node in relevant_nodes_old)
    partial_llh_new = sum(node.local_log_likelihood_contribution() for node in relevant_nodes_new)

    num_params_old = sum(1 for node in relevant_nodes_old for _ in node.transition_iterator())
    num_params_new = sum(1 for node in relevant_nodes_new for _ in node.transition_iterator())

    return partial_llh_old - partial_llh_new, num_params_old - num_params_new

def lower_threshold(value, thresh):
    if isinstance(value, Callable): # can be used to wrap score functions
        def fun(*args, **kwargs):
            return lower_threshold(value(*args, **kwargs), thresh)
        return fun
    return value if thresh < value else False

def likelihood_ratio_global_score(alpha) -> GlobalScoreFunction:
    alpha = log(alpha)
    def log_chi2(x, k):
        return (k/2 - 1) * log(x) - (x/2) * 1 - (k/2) * log(2) - lgamma(k/2)

    def score_fun(part : Dict[Node, Node]) :
        llh_diff, param_diff = differential_info(part)
        score = log_chi2(2*(llh_diff), param_diff)
        return lower_threshold(score, alpha) # Not entirely sure if implemented correctly
    return score_fun

def AIC_global_score(alpha = 0) -> GlobalScoreFunction:
    def score(part : Dict[Node, Node]) :
        llh_diff, param_diff = differential_info(part)
        return lower_threshold(param_diff - llh_diff, alpha)
    return score

def EDSM_global_score(min_evidence = -1) -> GlobalScoreFunction:
    def score(part : Dict[Node, Node]):
        total_evidence = 0
        for old_node, new_node in part.items():
            for in_sym, trans_old in old_node.transitions.items():
                trans_new = new_node.transitions.get(in_sym)
                if not trans_new:
                    continue
                for out_sym, trans_info in trans_old.items():
                    if out_sym in trans_new:
                        total_evidence += trans_info.count
        return lower_threshold(total_evidence, min_evidence)
    return score
