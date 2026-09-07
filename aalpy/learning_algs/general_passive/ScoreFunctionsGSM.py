# Score/compatibility function building blocks used to guide the general passive
# state-merging algorithm (local compatibility checks and global merge scores).
from collections import deque
from collections.abc import Callable, Iterable
from functools import total_ordering
from math import sqrt, log
from typing import Any

from aalpy.learning_algs.general_passive.GsmNode import GsmNode, intersection_iterator, union_iterator, CountData
from aalpy.learning_algs.general_passive.DataHandler import ShadowPTAData, CountOnPTAData

LocalCompatibilityFunction = Callable[[GsmNode, GsmNode], bool | None]
ScoreFunction = Callable[[dict[GsmNode, GsmNode]], Any]
AggregationFunction = Callable[[Iterable], Any]


class SpecialScores:
    @total_ordering
    class _SpecialScore:
        def __init__(self, ideal: bool):
            self.ideal = ideal

        def __lt__(self, other):
            return not self.ideal

        def __bool__(self):
            return self.ideal

    ImmediateAccept = _SpecialScore(True)
    ImmediateReject = _SpecialScore(False)
    NoScore = None

class ScoreCalculation:
    """Bundles a local compatibility check and a global score function used during state merging."""

    def __init__(self, local_compatibility: LocalCompatibilityFunction = None,
                 score_function: ScoreFunction = None) -> None:
        """
        Create a score calculation, optionally overriding the default (accept-everything) behavior.

        :param LocalCompatibilityFunction local_compatibility: Function determining local compatibility of two nodes.
        :param ScoreFunction score_function: Function computing the score of a full merge partition.
        """
        # This is a hack that gives a simple implementation where we can easily
        # - determine whether the default is overridden (for optimization)
        # - override behavior in a functional way by providing the functions as arguments (no extra class)
        # - override behavior in a stateful way by implementing a new class that provides `local_compatibility` and / or `score_function` methods
        if not hasattr(self, "local_compatibility"):
            self.local_compatibility: LocalCompatibilityFunction = local_compatibility or self.default_local_compatibility
        if not hasattr(self, "score_function"):
            self.score_function: ScoreFunction = score_function or self.default_score_function

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        """
        Callback at the beginning of the evaluation of a merge candidate.

        :param GsmNode red: GsmNode representing the red node of the merge candidate.
        :param GsmNode blue: GsmNode representing the blue node of the merge candidate.
        :param bool first_pass: Whether this is the first pass (in which the partitioning is only partially constructed)
          or the second pass (in which the partitioning is completed)
        :return: Either an early score for the merge candidate or `None`.
        """
        return None

    def promotion_score(self, promotion_candidate: GsmNode) -> Any:
        """
        Computes the score of a promotion candidate. By default, promotion candidates are immediately accepted. Override
        this function to implement promotion scoring.

        :param GsmNode promotion_candidate: GsmNode which is to be promoted.
        :return Any: The score of the promotion candidate. Default is ImmediateAccept.
        """
        return SpecialScores.ImmediateAccept

    @staticmethod
    def default_local_compatibility(a: GsmNode, b: GsmNode) -> bool:
        """
        Default local compatibility check: always compatible.

        :param GsmNode a: First node.
        :param GsmNode b: Second node.
        :return bool: Always True.
        """
        return True

    @staticmethod
    def default_score_function(part: dict[GsmNode, GsmNode]) -> Any:
        """
        Default score function: any partition is acceptable.

        :param dict[GsmNode, GsmNode] part: Mapping of original nodes to their merged partition representative.
        :return Any: Always accept.
        """
        return SpecialScores.ImmediateAccept

    def has_score_function(self) -> bool:
        """
        Check whether a non-default score function is configured.

        :return bool: True if score_function was overridden.
        """
        return self.score_function is not self.default_score_function


def hoeffding_compatibility(eps: float, compare_original: bool = True) -> LocalCompatibilityFunction:
    """
    Build a local compatibility function based on the Hoeffding bound over output distributions.

    :param float eps: Confidence parameter (smaller values are stricter).
    :param bool compare_original: Whether to compare counts from the original PTA rather than the current counts.
    :return LocalCompatibilityFunction: Function checking whether two nodes' output distributions are compatible.
    """
    eps_fact = sqrt(0.5 * log(2 / eps))

    def similar(a: GsmNode[CountData], b: GsmNode[CountData]) -> bool:
        # iterate over inputs that are common to both states
        if compare_original:
            a_dict = a.data.pta_count
            b_dict = b.data.pta_count
        else:
            a_dict = a.data.transition_count
            b_dict = b.data.transition_count

        for in_sym, a_trans, b_trans in intersection_iterator(a_dict, b_dict, True):
            # could create appropriate dict here
            a_total, b_total = (sum(trans.values()) for trans in (a_trans, b_trans))
            if a_total == 0 or b_total == 0:
                continue  # parameter combinations require this check
            threshold = eps_fact * (sqrt(1 / a_total) + sqrt(1 / b_total))
            # iterate over outputs that appear in either distribution
            for out_sym, ac, bc in union_iterator(a_trans, b_trans, 0):
                if abs(ac / a_total - bc / b_total) > threshold:
                    return False
        return True

    return similar

class SimpleFutureBasedScore(ScoreCalculation):
    """
    ScoreCalculation that checks local compatibility only on common futures (as in Alergia) and not during the
    construction of the partitioning. As long as no scoring (based on the partitioning) is used, this results in a
    significant speedup.
    """
    def __init__(self,
                 local_compatibility: LocalCompatibilityFunction = None,
                 score_function: ScoreFunction = None,
                 compatibility_on_pta = False,
                 depth_first = False,
                 ):
        """
        Create a new CheckFutureScore instance.

        :param LocalCompatibilityFunction local_compatibility: Compatibility criterion used to check futures.
        :param ScoreFunction score_function: The score function to rank merge candidates. Values other than None (default)
          negate the speedup.
        :param bool compatibility_on_pta: Whether compatibility should be checked on the PTA or the partially merged automaton
        :param bool depth_first: Whether to traverse the implied merges DFS or BFS. Defaults to True (BFS).
        """
        super().__init__(local_compatibility, score_function)
        self.compatibility_on_pta = compatibility_on_pta
        self.depth_first = depth_first

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        if self.compatibility_on_pta and not isinstance(red.data, ShadowPTAData):
            raise TypeError("compatibility_on_pta is set but no PTA data is available")

        q: deque[tuple[GsmNode, GsmNode]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft

        while len(q) != 0:
            red, blue = pop()

            local_compatibility = self.local_compatibility(red, blue)
            if local_compatibility is False:
                return SpecialScores.ImmediateReject
            if local_compatibility is None:
                continue

            if self.compatibility_on_pta:
                red_data: ShadowPTAData = red.data
                blue_data: ShadowPTAData = blue.data
                for in_sym, red_trans, blue_trans in intersection_iterator(red_data.shadow_pta, blue_data.shadow_pta, True):
                    for out_sym, red_child, blue_child in intersection_iterator(red_trans, blue_trans):
                        q.append((red_child,blue_child))
            else:
                for in_sym, red_trans, blue_trans in intersection_iterator(red.transitions, blue.transitions, True):
                    for out_sym, red_child, blue_child in intersection_iterator(red_trans, blue_trans):
                        q.append((red_child, blue_child))

        if self.has_score_function():
            return SpecialScores.NoScore
        return SpecialScores.ImmediateAccept


class ScoreIOAlergiaWithEDSM(SimpleFutureBasedScore):
    def __init__(self, eps: float, compat_on_pta: bool, compat_on_pta_data: bool, edsm: bool):
        self.compat = hoeffding_compatibility(eps, compat_on_pta_data)
        SimpleFutureBasedScore.__init__(self, None, compatibility_on_pta=compat_on_pta)
        self.edsm = edsm
        self.score = None

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        self.score = 0
        verdict = super().initialize_merge(red, blue, first_pass)
        if self.edsm is False or verdict is SpecialScores.ImmediateReject:
            return verdict
        return self.score

    def local_compatibility(self, red: GsmNode, blue: GsmNode) -> float:
        self.score += 1
        return self.compat(red, blue)


class ScoreWithKTail(ScoreCalculation):
    """Applies k-Tails to a compatibility function: Compatibility is only evaluated up to a certain depth k."""

    def __init__(self, other_score: ScoreCalculation, k: int) -> None:
        """
        Wrap another score calculation, limiting local compatibility checks to depth k.

        :param ScoreCalculation other_score: Score calculation to delegate to within depth k.
        :param int k: Maximum depth (relative to the blue node's initial depth) at which compatibility is checked.
        """
        super().__init__(None, other_score.score_function)
        self.other_score = other_score
        self.k = k

        self.depth_offset = None

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        self.depth_offset = blue.get_prefix_length()
        return self.other_score.initialize_merge(red, blue, first_pass)

    def local_compatibility(self, a: GsmNode, b: GsmNode) -> bool | None:
        """
        Check local compatibility, treating nodes beyond depth k as automatically compatible.

        :param GsmNode a: First (red) node.
        :param GsmNode b: Second (blue) node, assumed to be tree-shaped.
        :return bool: True if compatible (or beyond depth k), False otherwise.
        """
        # assuming b is tree shaped.
        depth = b.get_prefix_length() - self.depth_offset
        if self.k <= depth:
            return None

        return self.other_score.local_compatibility(a, b)


class ScoreWithSinks(ScoreCalculation):
    """This class allows rejecting merge candidates based on additional criteria for the initial merge"""

    def __init__(self, other_score: ScoreCalculation, sink_cond: Callable[[GsmNode], bool],
                 allow_sink_merge: bool = True) -> None:
        """
        Wrap another score calculation, additionally rejecting merges involving "sink" nodes.

        :param ScoreCalculation other_score: Score calculation to delegate to.
        :param Callable[[GsmNode], bool] sink_cond: Predicate identifying sink nodes.
        :param bool allow_sink_merge: Whether merges between two sink nodes are allowed.
        """
        super().__init__(None, other_score.score_function)
        self.other_score = other_score
        self.sink_cond = sink_cond
        self.allow_sink_merge = allow_sink_merge

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        a_sink, b_sink = self.sink_cond(red), self.sink_cond(blue)
        if (a_sink or b_sink) and not (a_sink and b_sink and self.allow_sink_merge):
            return SpecialScores.ImmediateReject
        return self.other_score.initialize_merge(red, blue, first_pass)

    def local_compatibility(self, a: GsmNode, b: GsmNode) -> bool:
        """
        Check local compatibility, additionally applying the sink condition on the first call.

        :param GsmNode a: First (red) node.
        :param GsmNode b: Second (blue) node.
        :return bool: True if compatible according to the sink condition and the wrapped score calculation.
        """
        return self.other_score.local_compatibility(a, b)


class ScoreCombinator(ScoreCalculation):
    """
    This class is used to combine several scoring / compatibility mechanisms by aggregating the results of the
    individual methods in a user defined manner. It uses generator expressions to allow for short circuit evaluation.
    """

    def __init__(self, scores: list[ScoreCalculation], aggregate_compatibility: AggregationFunction = None,
                 aggregate_score: AggregationFunction = None) -> None:
        """
        Combine several score calculations into one.

        :param list[ScoreCalculation] scores: Score calculations to combine.
        :param AggregationFunction aggregate_compatibility: Function aggregating the individual compatibility results.
        :param AggregationFunction aggregate_score: Function aggregating the individual score results.
        """
        super().__init__()
        self.scores = scores
        self.aggregate_compatibility = aggregate_compatibility or self.default_aggregate_compatibility
        self.aggregate_score = aggregate_score or self.default_aggregate_score

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        scores = [score.initialize_merge(red, blue, first_pass) for score in self.scores]
        return self.aggregate_score(scores)

    def local_compatibility(self, a: GsmNode, b: GsmNode) -> Any:
        """
        Compute the aggregated local compatibility of two nodes over all combined score calculations.

        :param GsmNode a: First node.
        :param GsmNode b: Second node.
        :return Any: Aggregated compatibility result.
        """
        return self.aggregate_compatibility(score.local_compatibility(a, b) for score in self.scores)

    def score_function(self, part: dict[GsmNode, GsmNode]) -> Any:
        """
        Compute the aggregated score of a merge partition over all combined score calculations.

        :param dict[GsmNode, GsmNode] part: Mapping of original nodes to their merged partition representative.
        :return Any: Aggregated score result.
        """
        return self.aggregate_score(score.score_function(part) for score in self.scores)

    @staticmethod
    def default_aggregate_compatibility(compatibility_iterable: Iterable) -> Any:
        """
        Commits to the first value that is not inconclusive (== None). Accepts if in doubt.

        :param Iterable compatibility_iterable: Iterable of compatibility results.
        :return Any: The first non-None result, or True if all are None.
        """
        for compat in compatibility_iterable:
            if compat is None:
                continue
            return compat
        return True

    @staticmethod
    def default_aggregate_score(score_iterable: Iterable) -> list:
        """
        Default score aggregation: collect all scores into a list.

        :param Iterable score_iterable: Iterable of score results.
        :return list: List of the individual scores.
        """
        return list(score_iterable)


def local_to_global_compatibility(local_fun: LocalCompatibilityFunction) -> ScoreFunction:
    """
    Converts a local compatibility function to a global score function by evaluating the local compatibility for each of
    the new partitions with all nodes that make up that partition. One use case for this is to evaluate a local score
    function after the partitions are complete. The order of arguments for the local compatibility function is
    partition, original.

    :param LocalCompatibilityFunction local_fun: Local compatibility function to lift to a global score function.
    :return ScoreFunction: Global score function rejecting if any local check fails and greedily accepting otherwise.
    """

    def fun(part: dict[GsmNode, GsmNode]) -> Any:
        for old_node, new_node in part.items():
            if local_fun(new_node, old_node) is False:  # Follows local_fun(red, blue)
                return SpecialScores.ImmediateReject
        return SpecialScores.ImmediateAccept

    return fun


def transform_score(transform: Callable) -> Any:
    """
    Lifts an operation on a score value to score functions and ScoreCalculation objects. Intended as a decorator

    :param Callable transform: Function to apply to the (eventual) score value.
    :return Any: Decorated transformation applicable to a score, callable, or ScoreCalculation.
    """
    def fun(score: Any, *args) -> Any:
        if isinstance(score, Callable):
            return lambda partitioning: transform(score(partitioning), *args)
        if isinstance(score, ScoreCalculation):
            original_score_function = score.score_function
            score.score_function = lambda partitioning: transform(original_score_function(partitioning), *args)
            return score
        return transform(score, *args)

    return fun

@transform_score
def greedy_score(score: Any) -> Any:
    """
    Transform a score into a greedy (boolean) score: accept anything but a False/reject result.

    :param Any score: A plain value, callable score function, or ScoreCalculation instance.
    :return Any: The transformed score, callable, or ScoreCalculation.
    """
    should_accept = score is not False and score is not SpecialScores.ImmediateReject
    return SpecialScores.ImmediateAccept if should_accept else SpecialScores.ImmediateReject


@transform_score
def lower_threshold(score: Any, thresh: Any) -> Any:
    """
    Transform a score so that it is rejected (False) unless it exceeds a threshold.

    :param Any score: A plain value, callable score function, or ScoreCalculation instance.
    :param Any thresh: Threshold the score must exceed to be accepted.
    :return Any: The transformed score, callable, or ScoreCalculation.
    """
    return score if thresh <= score else SpecialScores.ImmediateReject


def differential_info(part: dict[GsmNode[CountData], GsmNode[CountData]]) -> tuple[float, int]:
    """
    Compute the change in log-likelihood and number of parameters caused by a merge partition.

    :param dict[GsmNode, GsmNode] part: Mapping of original nodes to their merged partition representative.
    :return tuple[float, int]: (log-likelihood difference, parameter count difference) between old and new nodes.
    """
    relevant_nodes_old = list(part.keys())
    relevant_nodes_new = set(part.values())

    partial_llh_old = sum(node.data.local_log_likelihood_contribution() for node in relevant_nodes_old)
    partial_llh_new = sum(node.data.local_log_likelihood_contribution() for node in relevant_nodes_new)

    num_params_old = sum(1 for node in relevant_nodes_old for _ in node.child_iterator())
    num_params_new = sum(1 for node in relevant_nodes_new for _ in node.child_iterator())

    return partial_llh_old - partial_llh_new, num_params_old - num_params_new


def AIC_score(alpha: float = 0) -> ScoreFunction:
    """
    Build a score function based on the Akaike information criterion (AIC).

    :param float alpha: Threshold applied to the AIC-based score.
    :return ScoreFunction: Score function computing the AIC-based score of a merge partition.
    """
    def score(part: dict[GsmNode, GsmNode]) -> Any:
        llh_diff, param_diff = differential_info(part)
        return lower_threshold(param_diff - llh_diff, alpha)

    return score


def EDSM_frequency_score(min_evidence: int = 0) -> ScoreFunction:
    """
    Build a score function counting the total evidence (transition count) contradicted by a merge.

    :param int min_evidence: Minimum evidence required for the merge to be accepted.
    :return ScoreFunction: Score function computing the total contradicting evidence of a merge partition.
    """
    def score(part: dict[GsmNode[CountData], GsmNode[CountData]]) -> Any:
        total_evidence = 0
        for old_node, new_node in part.items():
            for in_sym, old_trans, new_trans_new in intersection_iterator(old_node.data.transition_count, new_node.data.transition_count):
                for out_sym, old_count, new_count in intersection_iterator(old_trans, new_trans_new):
                    if old_count != new_count:
                        total_evidence += old_count
        return lower_threshold(total_evidence, min_evidence)

    return score


def EDSM_score(min_evidence: int = -1) -> ScoreFunction:
    """
    Build the classic Evidence Driven State Merging (EDSM) score function.

    :param int min_evidence: Minimum number of merged states required for the merge to be accepted.
    :return ScoreFunction: Score function computing the number of merged states minus the number of partitions.
    """
    def score(part: dict[GsmNode, GsmNode]) -> Any:
        nr_partitions = len(set(part.values()))
        nr_merged = len(part)
        return lower_threshold(nr_merged - nr_partitions, min_evidence)

    return score
