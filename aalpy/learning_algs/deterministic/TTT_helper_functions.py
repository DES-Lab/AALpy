import random

from aalpy.automata import Dfa
from aalpy.base import SUL
from aalpy.learning_algs.deterministic.DiscriminationTree import DTStateNode, DiscriminationTree
from aalpy.learning_algs.deterministic.TTTHypothesis import TTTHypothesisState, TTTHypothesis


def link(dtree_node: DTStateNode, hypothesis_state: TTTHypothesisState):
    dtree_node.hyp_state = hypothesis_state
    hypothesis_state.dtree_node = dtree_node
    if dtree_node.parent.parent is None and dtree_node.path_to_node == "true":
        hypothesis_state.is_accepting = True


def close_transitions(dtree: DiscriminationTree, hypothesis: TTTHypothesis):
    '''
    Close all open transitions in the hypothesis.
    Transition targets are determined by sifting:
    given a state q ∈ HYP_STATES identified by a prefix p,
    it’s x-Successor (x ∈ Σ) is determined by sifting p•x into the dtree

    Args:
        dtree: the DiscriminationTree
        hypothesis: the TTTHypothesis

    '''

    def _close_transitions():
        open_transitions = hypothesis.get_open_transitions()
        new_states = []
        for start_state, letter in open_transitions:
            dtree_node = dtree.sift((*start_state.prefix, letter))
            if not dtree_node.hyp_state:
                new_state = hypothesis.new_state()
                link(dtree_node, new_state)
                new_states.append(new_state)
            start_state.transitions[letter] = dtree_node.hyp_state
        return new_states

    new_states = _close_transitions()
    while new_states:
        new_states = _close_transitions()


def rs_split_cex(cex: list, hyp: Dfa, sul: SUL, alphabet):
    '''
    find u,a,v such that δA(q0,[u]_(HYP)av) != δA(q0,[ua]_(HYP)v)
    δA(q0,x) == sul.query(x)
    [u]_(HYP) == Word u leads to state x in HYP, state x has prefix p => p


    Args:
        cex: list, counterexample
        hyp: Dfa, hypothesis
        sul: SUL, system under learning
        alphabet: the alphabet

    Returns:

    '''
    # TODO replace this function with the already implemented rs_cex_processing:
    # ca.:
    # v = rs_cex_processing(cex)
    # a = cex[len(cex)-len(v)-1]
    # u = cex[:len(cex)-len(v)-1]

    assert len(cex) > 1

    def _split_cex(u_a_split, a_v_split):
        u = cex[:u_a_split] or [None]
        a = cex[u_a_split:a_v_split]
        v = cex[a_v_split:] or [None]
        return(u,a,v)

    def _leads_to_state(word):
        hyp.reset_to_initial()
        hyp.execute_sequence(hyp.initial_state, word)
        return hyp.current_state

    def _get_random_split():
        u_a_split = random.randint(0, len(cex)-1)
        a_v_split = u_a_split + 1
        return u_a_split, a_v_split

    u_a_split = 0
    a_v_split = 1

    while True:
        u, a, v = _split_cex(u_a_split, a_v_split)

        u_state = _leads_to_state(u)
        ua_state = _leads_to_state((*u, *a))

        u_mq = sul.query((*u_state.prefix, *a, *v))[-1]
        ua_mq = sul.query((*ua_state.prefix, *v))[-1]

        if u_mq != ua_mq:
            old_out = ua_mq
            new_out = u_mq
            return u, a, v, u_state, ua_state, old_out, new_out
        else:
            u_a_split, a_v_split = _get_random_split()




