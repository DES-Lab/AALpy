# Score/compatibility function building blocks used to guide the general passive
# state-merging algorithm (local compatibility checks and global merge scores).
from abc import ABC
from collections import deque
from collections.abc import Callable, Iterable
from functools import total_ordering
from math import sqrt, log
from typing import Any

from aalpy.learning_algs.general_passive.GsmNode import GsmNode, intersection_iterator, union_iterator, CountData
from aalpy.learning_algs.general_passive.AssociatedData import ShadowPTAData

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

class ScoreCalculation(ABC):
    """Bundles a local compatibility check and a global score function used during state merging."""

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

    def local_compatibility(self, a: GsmNode, b: GsmNode) -> bool | None:
        """
        Computes whether two `GsmNode` are locally compatible. It is called during partition construction. If not overridden,
        any two nodes are considered compatible, unless `output_behavior` is set to `"moore"`. Overriding allows rejecting
        a merge candidate early, without having to construct the full partitioning.

        :param  GsmNode a: The node corresponding to the current (=partial) partition of a node.
        :param  GsmNode b: The node to be merged into the partition.
        :return bool | None: Whether the two `GsmNode` are locally compatible. Returns `None` if no further descendants
          need to be considered to assess the final verdict / score.
        """
        return True

    def score_function(self, part: dict[GsmNode, GsmNode]) -> Any:
        """
        Computes the score of a merge candidate based on the partitioning resulting from implied merges (determinization)
        starting from the original merge candidate.

        :param  dict[GsmNode, GsmNode] part: Mapping of original nodes to their merged partition representative.
        :return Any: The score of the merge candidate. Special values are:
          - SpecialScores.ImmediateAccept: the merge candidate should be merged without considering others.
          - SpecialScores.ImmediateReject: the merge candidate should not be considered.
          Default is immediate acceptance.
        """
        return SpecialScores.ImmediateAccept

    def promotion_score(self, promotion_candidate: GsmNode) -> Any:
        """
        Computes the score of a promotion candidate. By default, promotion candidates are immediately accepted. Override
        this function to implement promotion scoring.

        :param GsmNode promotion_candidate: GsmNode which is to be promoted.
        :return Any: The score of the promotion candidate. Default is ImmediateAccept.
        """
        return SpecialScores.ImmediateAccept

    def has_local_compatibility(self) -> bool:
        """
        Check whether a non-default local compatibility is configured.

        :return bool: True if local_compatibility was overridden.
        """
        return self.__class__.local_compatibility is not ScoreCalculation.local_compatibility

    def has_score_function(self) -> bool:
        """
        Check whether a non-default score function is configured.

        :return bool: True if score_function was overridden.
        """
        return self.__class__.score_function is not ScoreCalculation.score_function


class SimpleScoreCalculation(ScoreCalculation):
    def __init__(self, local_compatibility: LocalCompatibilityFunction = None, score_function: ScoreFunction = None) -> None:
        self.local_compatibility = local_compatibility or self.local_compatibility
        self._has_local_compatibility = local_compatibility is not None
        self.score_function = score_function or self.score_function
        self._has_score_function = score_function is not None

    def has_local_compatibility(self) -> bool:
        return self._has_local_compatibility

    def has_score_function(self) -> bool:
        return self._has_score_function


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


class SimpleFutureBasedCompatibility(ScoreCalculation):
    """
    ScoreCalculation without scoring that checks local compatibility only on common futures (as in Alergia) and not
    during the construction of the partitioning. This avoids the need to construct the partitioning in a reversible manner,
    which results in a significant speedup.
    """
    def __init__(self,
                 compatibility_on_pta = False,
                 depth_first = False,
                 local_compatibility: LocalCompatibilityFunction = None,
                 ):
        """
        Create a new CheckFutureScore instance.

        :param bool compatibility_on_pta: Whether compatibility should be checked on the PTA or the partially merged automaton
        :param bool depth_first: Whether to traverse the implied merges DFS or BFS. Defaults to True (BFS).
        :param LocalCompatibilityFunction local_compatibility: Compatibility criterion used to check futures.
        """
        if local_compatibility:
            if self.has_local_compatibility():
                raise ValueError("External local compatibility is provided, but the class already defines a local compatibility criterion.")
            self.local_compatibility = local_compatibility or self.local_compatibility
        self.compatibility_on_pta = compatibility_on_pta
        self.depth_first = depth_first

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        if not first_pass:
            return

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

        return SpecialScores.ImmediateAccept


class ScoreIOAlergiaWithEDSM(SimpleFutureBasedCompatibility):
    def __init__(self, eps: float, compat_on_pta: bool, compat_on_pta_data: bool, edsm: bool):
        self.compat = hoeffding_compatibility(eps, compat_on_pta_data)
        SimpleFutureBasedCompatibility.__init__(self, compatibility_on_pta=compat_on_pta)
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


class WrappingScore(ScoreCalculation, ABC):
    """Baseclass for wrapping `ScoreCalculation` objects with minor changes."""

    def __init__(self, wrapped: ScoreCalculation):
        self.wrapped = wrapped
        # TODO could detect overrides and hardlink to methods of wrapped score otherwise. see below.
        # if not hasattr(self, "initialized_merge"):
        #     self.initialized_merge = wrapped.initialize_merge

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        return self.wrapped.initialize_merge(red, blue, first_pass)

    def local_compatibility(self, red: GsmNode, blue: GsmNode) -> bool | None:
        return self.wrapped.local_compatibility(red, blue)

    def score_function(self, part: dict[GsmNode, GsmNode]) -> Any:
        return self.wrapped.score_function(part)

    def promotion_score(self, promotion_candidate: GsmNode) -> Any:
        return self.wrapped.promotion_score(promotion_candidate)

    def has_local_compatibility(self) -> bool:
        return self.wrapped.has_local_compatibility()

    def has_score_function(self) -> bool:
        return self.wrapped.has_score_function()


class ScoreWithKTail(WrappingScore):
    """Applies k-Tails to a compatibility function: Compatibility is only evaluated up to a certain depth k."""

    def __init__(self, wrapped: ScoreCalculation, k: int) -> None:
        """
        Wrap another score calculation, limiting local compatibility checks to depth k.

        :param ScoreCalculation wrapped: Score calculation to delegate to within depth k.
        :param int k: Maximum depth (relative to the blue node's initial depth) at which compatibility is checked.
        """
        super().__init__(wrapped)
        self.k = k

        self.depth_offset = None

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        self.depth_offset = blue.get_prefix_length()
        return self.wrapped.initialize_merge(red, blue, first_pass)

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

        return self.wrapped.local_compatibility(a, b)


class ScoreWithSinks(WrappingScore):
    """This class allows rejecting merge candidates based on additional criteria for the initial merge"""

    def __init__(self, wrapped: ScoreCalculation, sink_cond: Callable[[GsmNode], bool],
                 allow_sink_merge: bool = True) -> None:
        """
        Wrapped score calculation, additionally rejecting merges involving "sink" nodes.

        :param ScoreCalculation wrapped: Score calculation to delegate to.
        :param Callable[[GsmNode], bool] sink_cond: Predicate identifying sink nodes.
        :param bool allow_sink_merge: Whether merges between two sink nodes are allowed.
        """
        super().__init__(wrapped)
        self.sink_cond = sink_cond
        self.allow_sink_merge = allow_sink_merge

    def initialize_merge(self, red: GsmNode, blue: GsmNode, first_pass: bool) -> Any:
        a_sink, b_sink = self.sink_cond(red), self.sink_cond(blue)
        if (a_sink or b_sink) and not (a_sink and b_sink and self.allow_sink_merge):
            return SpecialScores.ImmediateReject
        return self.wrapped.initialize_merge(red, blue, first_pass)


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
        Compute the aggregated score of a merge candidate over all combined score calculations.

        :param dict[GsmNode, GsmNode] part: Mapping of original nodes to their merged partition representative.
        :return Any: Aggregated score result.
        """
        return self.aggregate_score(score.score_function(part) for score in self.scores)

    def promotion_score(self, promotion_candidate: GsmNode) -> Any:
        """
        Compute the aggregated score of a promotion over all combined score calculations.

        :param GsmNode promotion_candidate: Node to be promoted.
        :return Any: Aggregated score result.
        """
        return self.aggregate_score(score.promotion_score(promotion_candidate) for score in self.scores)

    @staticmethod
    def default_aggregate_compatibility(compatibility_iterable: Iterable) -> Any:
        """
        Returns the least permissive among provided values (False < True < None).

        :param Iterable compatibility_iterable: Iterable of compatibility results.
        :return Any: The first non-None result, or True if all are None.
        """
        highest_value = None
        for compat in compatibility_iterable:
            if compat is False:
                return False
            if compat is True:
                highest_value = True
        return highest_value

    @staticmethod
    def default_aggregate_score(score_iterable: Iterable) -> Any:
        """
        Default score aggregation: collect all scores into a list, unless a special value decides the outcome.
        Rejection wins over anything else and a single undecided score leaves the aggregate undecided, whereas
        acceptance has to be unanimous, since a list mixing special values is not a meaningful score.

        :param Iterable score_iterable: Iterable of score results.
        :return Any: The deciding special value, or the list of the individual scores.
        """
        scores = list(score_iterable)
        for special in (SpecialScores.ImmediateReject, SpecialScores.NoScore):
            if any(score is special for score in scores):
                return special
        if scores and all(score is SpecialScores.ImmediateAccept for score in scores):
            return SpecialScores.ImmediateAccept
        return scores


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


def score_transformation(transform: Callable) -> Any:
    """
    Lifts an operation on a score value to score functions and ScoreCalculation objects. Intended as a decorator

    :param Callable transform: Function to apply to the (eventual) score value.
    :return Any: Decorated transformation applicable to a score, callable, or ScoreCalculation.
    """
    def score_function(score: Any, *transformation_args, **transformation_kwargs) -> Any:
        if isinstance(score, Callable):
            return lambda partitioning: transform(score(partitioning), *transformation_args, **transformation_kwargs)
        if isinstance(score, ScoreCalculation):
            original_score_function = score.score_function
            score.score_function = lambda partitioning: transform(original_score_function(partitioning), *transformation_args, **transformation_kwargs)
            return score
        return transform(score, *transformation_args, **transformation_kwargs)

    return score_function

@score_transformation
def greedy_score(score: Any) -> Any:
    """
    Transform a score into a greedy (boolean) score: accept anything but a False/reject result.

    :param Any score: A plain value, callable score function, or ScoreCalculation instance.
    :return Any: The transformed score, callable, or ScoreCalculation.
    """
    should_accept = score is not False and score is not SpecialScores.ImmediateReject
    return SpecialScores.ImmediateAccept if should_accept else SpecialScores.ImmediateReject


@score_transformation
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
