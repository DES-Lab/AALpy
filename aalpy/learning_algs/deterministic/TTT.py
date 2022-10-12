import time

from aalpy.base import Oracle, SUL
from aalpy.utils.HelperFunctions import extend_set, print_learning_info, print_observation_table, all_prefixes
from .CounterExampleProcessing import longest_prefix_cex_processing, rs_cex_processing
from .DiscriminationTree import DiscriminationTree, DTStateNode, DTDiscriminatorNode
from .ObservationTable import ObservationTable
from .TTTHypothesis import TTTHypothesis
from .TTT_helper_functions import link, close_transitions, rs_split_cex
from ...SULs import DfaSUL
from ...base.SUL import CacheSUL
from aalpy.automata import Dfa

counterexample_processing_strategy = [None, 'rs', 'longest_prefix']
closedness_options = ['prefix', 'suffix']
print_options = [0, 1, 2, 3]


def run_TTT(alphabet: list, sul: SUL, eq_oracle: Oracle, automaton_type='dfa',
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

    # initial dtree: root node with discriminator None (i.e. the empty word, "epsilon") and empty "True" and "False" children
    dtree = DiscriminationTree(alphabet=alphabet, sul=sul)
    hypothesis = TTTHypothesis(alphabet=alphabet)

    # sift the empty word into the dtree
    init_node = dtree.sift(word=(None,))

    # connect dtree node and hypothesis state
    link(init_node, hypothesis.initial_state)

    # close all transitions
    close_transitions(dtree, hypothesis)

    eq_query_start = time.time()
    cex = eq_oracle.find_cex(hypothesis.dfa)
    # cex = ['b', 'a', 'a']
    eq_query_time += time.time() - eq_query_start

    while cex is not None:
        print(f'Sent hypothesis {hypothesis.dfa}, received counterexample {cex}')

        # rs_cex_processing(sul, cex, hypothesis.dfa)
        u, a, v, u_state, ua_state, old_out, new_out = rs_split_cex(cex, hypothesis.dfa, sul, alphabet)
        print(f'Split counterexample: {u=} {a=} {v=} {u_state=} {ua_state=} {old_out=} {new_out=}')

        a = a[0]
        # ua_state needs to be split
        # in hypothesis, introduce a new state with access sequence [u]_HYP*a (u_state.prefix)*a
        new_state = hypothesis.new_state()
        assert u_state.transitions[a] == ua_state
        old_state = u_state.transitions[a]
        u_state.transitions[a] = new_state

        # in dtree
        old_dtree_node = ua_state.dtree_node
        assert isinstance(old_dtree_node, DTStateNode)
        new_dtree_node = DTDiscriminatorNode(discriminator=v,
                                             parent=old_dtree_node.parent,
                                             path_to_node=old_dtree_node.path_to_node)
        if old_out:
            old_node = u_state.dtree_node.true_child = DTStateNode(parent=new_dtree_node,
                                                                   path_to_node=True,
                                                                   prefix=old_dtree_node.prefix)
            new_node = u_state.dtree_node.false_child = DTStateNode(parent=new_dtree_node,
                                                                    path_to_node=False,
                                                                    prefix=(*u_state.prefix, a)) #prefix is incorrect prolly
        else:
            old_node = u_state.dtree_node.false_child = DTStateNode(parent=new_dtree_node,
                                                                    path_to_node=True,
                                                                    prefix=old_dtree_node.prefix)
            new_node = u_state.dtree_node.true_child = DTStateNode(parent=new_dtree_node,
                                                                   path_to_node=False,
                                                                   prefix=(*u_state.prefix, a))


        link(old_node, old_state)
        link(new_node, new_state)

        close_transitions(dtree, hypothesis)

        eq_query_start = time.time()
        cex = eq_oracle.find_cex(hypothesis.dfa)
        # cex = ['b', 'a', 'a']
        eq_query_time += time.time() - eq_query_start

        # src_state = hypothesis.initial_state
        #
        # hyp_dfa = hypothesis.dfa
        # hyp_dfa.execute_sequence(hyp_dfa.initial_state, u)
        # pred_state = hyp_dfa.current_state
        #
        # hyp_dfa.reset_to_initial()
        # hyp_dfa.execute_sequence(hyp_dfa.initial_state, (*u, *a))
        # succ_state = hyp_dfa.current_state
        #
        # hyp_dfa.reset_to_initial()
        # hyp_dfa.execute_sequence(pred_state, a)
        # assert hyp_dfa.current_state == succ_state
        #
        # trans = (pred_state, a) #should not be in tree...?
        #
        # # split_state(trans, v, old_out, new_out)
        # dt_node = trans[0].dtree_node # or just pred_state.dtree_node
        # assert isinstance(dt_node, DTStateNode)
        # old_state = pred_state

        # create transition in HYP from pred_state with a
