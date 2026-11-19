# L# active learning algorithm based on apartness and an observation tree.
import time

from aalpy.base import Automaton, Oracle, SUL
from aalpy.utils.HelperFunctions import print_learning_info
from .ObservationTree import ObservationTree
from ...base.SUL import CacheSUL


def run_Lsharp(alphabet: list, sul: SUL, eq_oracle: Oracle, automaton_type: str,
               extension_rule: str | None = 'SepSeq', separation_rule: str = "ADS", samples: list | None = None,
               max_learning_rounds: int | None = None, cache_and_non_det_check: bool = True,
               return_data: bool = False, print_level: int = 2) -> Automaton | tuple[Automaton, dict]:
    """
    Based on ''A New Approach for Active Automata Learning Based on Apartness'' from Vaandrager, Garhewal, Rot and Wissmann.
    and ''L# for DFAs'' from Vaandrager, Sanders.

    The algorithm learns a DFA/Moore machine/Mealy machine using apartness and an observation tree.

    :param list alphabet: Input alphabet.
    :param SUL sul: System under learning.
    :param Oracle eq_oracle: Equivalence oracle.
    :param str automaton_type: Type of automaton to be learned. Either 'dfa', 'mealy' or 'moore'.
    :param str | None extension_rule: Strategy used during the extension rule. Options: None, "SepSeq"
        (default) and "ADS".
    :param str separation_rule: Strategy used during the extension rule. Options: "SepSeq" (default) and "ADS".
    :param list | None samples: Input output traces provided to the learning algorithm. They are added to cache
        and could reduce total interaction with the system. Syntax: list of [(input_sequence, output_sequence)]
        or None.
    :param int | None max_learning_rounds: Number of learning rounds after which learning will terminate
        (Default value = None).
    :param bool cache_and_non_det_check: Use caching and non-determinism checks (Default value = True).
    :param bool return_data: If True, a map containing all information (runtime/#queries/#steps) will be returned
        (Default value = False).
    :param int print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 -
        educational/debug (Default value = 2).
    :return Automaton | tuple[Automaton, dict]: Automaton of type automaton_type (or a tuple of the automaton and
        a dict containing all information about learning if 'return_data' is True).
    """
    assert extension_rule in {None, "SepSeq", "ADS"}
    assert separation_rule in {"SepSeq", "ADS"}

    if cache_and_non_det_check or samples is not None:
        # Wrap the sul in the CacheSUL, so that all steps/queries are cached
        sul = CacheSUL(sul)
        eq_oracle.sul = sul

        if samples:
            for input_seq, output_seq in samples:
                sul.cache.add_to_cache(input_seq, output_seq)

    ob_tree = ObservationTree(alphabet, sul, automaton_type, extension_rule, separation_rule)
    start_time = time.time()

    eq_query_time = 0
    learning_rounds = 0
    hypothesis = None

    while True:
        learning_rounds += 1
        if max_learning_rounds and learning_rounds == max_learning_rounds:
            break

        # Building the hypothesis
        hypothesis = ob_tree.build_hypothesis()

        if print_level > 1:
            print(f'Hypothesis {learning_rounds}: {hypothesis.size} states.')
        if print_level == 3:
            print(hypothesis)

        # Pose Equivalence Query
        eq_query_start = time.time()
        cex = eq_oracle.find_cex(hypothesis)
        eq_query_time += time.time() - eq_query_start

        if print_level > 2:
            print(f'Counterexample: {cex}')

        if cex is None:
            break

        # Process the counterexample and start a new learning round
        cex_outputs = sul.query(cex)
        ob_tree.process_counter_example(hypothesis, cex, cex_outputs)

    total_time = round(time.time() - start_time, 2)
    eq_query_time = round(eq_query_time, 2)
    learning_time = round(total_time - eq_query_time, 2)

    info = {
        'learning_rounds': learning_rounds,
        'automaton_size': hypothesis.size,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': eq_oracle.num_queries,
        'steps_eq_oracle': eq_oracle.num_steps,
        'learning_time': learning_time,
        'eq_oracle_time': eq_query_time,
        'total_time': total_time,
        'cache_saved': sul.num_cached_queries,
    }

    if print_level > 0:
        print_learning_info(info)

    if return_data:
        return hypothesis, info

    return hypothesis
