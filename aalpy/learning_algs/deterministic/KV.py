import time

from aalpy.base import Oracle, SUL
from aalpy.utils.HelperFunctions import extend_set, print_learning_info, print_observation_table, all_prefixes
from .ClassificationTree import ClassificationTree, CTInternalNode, CTLeafNode
from .CounterExampleProcessing import longest_prefix_cex_processing, rs_cex_processing
from .DiscriminationTree import DiscriminationTree, DTStateNode, DTDiscriminatorNode
from .KV_helpers import state_name_gen
from .ObservationTable import ObservationTable
from .TTTHypothesis import TTTHypothesis
from .TTT_helper_functions import link, close_transitions, rs_split_cex
from ...SULs import DfaSUL
from ...base.SUL import CacheSUL
from aalpy.automata import Dfa, DfaState

counterexample_processing_strategy = [None, 'rs', 'longest_prefix']
closedness_options = ['prefix', 'suffix']
print_options = [0, 1, 2, 3]

def run_KV(alphabet: list, sul: SUL, eq_oracle: Oracle, automaton_type='dfa',
           max_learning_rounds=None, cache_and_non_det_check=True, return_data=False, print_level=2):
    """
    Executes TTT algorithm.

    Args:

        alphabet: input alphabet

        sul: system under learning

        eq_oracle: equivalence oracle

        automaton_type: type of automaton to be learned. Currently only 'dfa' supported

        max_learning_rounds: number of learning rounds after which learning will terminate (Default value = None)

        cache_and_non_det_check: Use caching and non-determinism checks (Default value = True)

        return_data: if True, a map containing all information(runtime/#queries/#steps) will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

    Returns:

        automaton of type automaton_type (dict containing all information about learning if 'return_data' is True)

    """

    assert print_level in print_options
    assert automaton_type == 'dfa'
    assert isinstance(sul, DfaSUL)

    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0

    # name_gen = state_name_gen('s')

    # Do a membership query on the empty string to determine whether
    # the start state of the SUL is accepting or rejecting
    empty_string_mq = sul.query((None,))[-1]

    # Construct a hypothesis automaton that consists simply of this
    # single (accepting or rejecting) state with self-loops for
    # all transitions.
    initial_state = DfaState(state_id=(None,),
                             is_accepting=empty_string_mq)
    for a in alphabet:
        initial_state.transitions[a] = initial_state

    hypothesis = Dfa(initial_state=initial_state,
                     states=[initial_state])

    # Perform an equivalence query on this automaton
    eq_query_start = time.time()
    cex = tuple(eq_oracle.find_cex(hypothesis))
    eq_query_time += time.time() - eq_query_start

    # initialise the classification tree to have a root
    # labeled with the empty word as the distinguishing string
    # and two leafs labeled with access strings cex and empty word
    ctree = ClassificationTree(alphabet=alphabet,
                               sul=sul,
                               cex=cex,
                               empty_is_true=empty_string_mq)

    while True:
        hypothesis = ctree.gen_hypothesis()

        # Perform an equivalence query on this automaton
        eq_query_start = time.time()
        cex = tuple(eq_oracle.find_cex(hypothesis))
        eq_query_time += time.time() - eq_query_start

        if cex is None:
            break

        cex_should_be = not hypothesis.execute_sequence(hypothesis.initial_state, cex)[-1]

        j = None
        for i in range(len(cex) + 1):
            s_i = ctree.sift(cex[:i] or (None,))
            hypothesis.execute_sequence(hypothesis.initial_state, cex[:i] or (None,))
            s_star_i = hypothesis.current_state.state_id
            if s_i != s_star_i:
                j = i
                break
        assert j is not None

        s_j_minus_1 = ctree.sift(cex[:j-1] or (None,))
        node_to_replace = ctree.leaf_nodes[s_j_minus_1]

        # TODO what should the distinguishing string be here?
        d_node = node_to_replace.parent.children[node_to_replace.path_to_node] = CTInternalNode(distinguishing_string=(*cex[:j],cex[j-1]),
                                                                                       parent=node_to_replace.parent)

        node_to_replace.parent = d_node
        # TODO is this access string correct? this already seems to exist
        new_node = CTLeafNode(access_string=cex[:j-1] or (None,),
                              parent=d_node,
                              tree=ctree)

        d_node.children[cex_should_be] = new_node
        d_node.children[not cex_should_be] = node_to_replace

    return hypothesis

