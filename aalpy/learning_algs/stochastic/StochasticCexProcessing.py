from aalpy.automata import Mdp
from aalpy.base import SUL


def stochastic_longest_prefix(cex, prefixes):
    """
    Counterexample processing based on Shabaz-Groz cex processing.

    Args:

        cex: counterexample
        prefixes: all prefixes in the observation table
    Returns:

        Single suffix.
    """
    prefixes = list(prefixes)
    prefixes.sort(key=len, reverse=True)

    trimmed_cex = None
    trimmed = False
    for p in prefixes:
        if p[1::2] == cex[:len(p)][1::2]:
            trimmed_cex = cex[len(p):]
            trimmed = True
            break

    trimmed_cex = trimmed_cex if trimmed else cex
    trimmed_cex = list(trimmed_cex)

    if not trimmed_cex:
        return ()

    # get all suffixes and return
    suffixes = [tuple(trimmed_cex[len(trimmed_cex) - i - 1:]) for i in range(0, len(trimmed_cex), 2)]

    # prefixes
    # need to pop 0 for MDP, for SMM remove the line
    # trimmed_cex.pop(0)
    # prefixes = [tuple(trimmed_cex[:i + 1]) for i in range(0, len(trimmed_cex), 2)]

    # TODO we could return all suffixes, remains as a option to be seen
    return [suffixes[-1]]


def stochastic_rs(sul: SUL, cex: tuple, hypothesis):
    """Riverst-Schapire counter example processing.

    Args:

        sul: system under learning
        cex: found counterexample
        hypothesis: hypothesis on which counterexample was found
    Returns:

        suffixes to be added to the E set

    """
    # cex_out = self.sul.query(tuple(cex))

    if isinstance(hypothesis, Mdp):
        cex = cex[1:]

    inputs = tuple(cex[::2])
    outputs = tuple(cex[1::2])
    # cex_out = self.teacher.sul.query(cex)

    lower = 1
    upper = len(inputs) - 2

    while True:
        hypothesis.reset_to_initial()
        mid = (lower + upper) // 2

        # arr[:n] -> first n values
        # arr[n:] -> last n values

        for i, o in zip(inputs[:mid], outputs[:mid]):
            hypothesis.step_to(i, o)

        s_bracket = hypothesis.current_state.prefix

        # prefix in hyp is reached

        prefix_inputs = s_bracket[1::2] if isinstance(hypothesis, Mdp) else s_bracket[::2]
        # prefix_outputs = s_bracket[0::2] if isinstance(hypothesis, Mdp) else s_bracket[1::2]

        not_same = False

        prefix_reached = False
        while not prefix_reached:
            hypothesis.reset_to_initial()
            sul.post()
            sul.pre()

            repeat = False
            for inp in prefix_inputs:
                o_sul = sul.step(inp)
                o_hyp = hypothesis.step_to(inp, o_sul)

                if o_hyp is None:
                    repeat = True
                    break

            prefix_reached = not repeat

        for inp in inputs[mid:]:

            o_sul = sul.step(inp)
            o_hyp = hypothesis.step_to(inp, o_sul)

            if o_hyp is None:
                not_same = True
                break

        if not not_same:
            lower = mid + 1
            if upper < lower:
                suffix = cex[(mid + 1) * 2:]
                break
        else:
            upper = mid - 1
            if upper < lower:
                suffix = cex[mid * 2:]
                break

    suffixes = [tuple(suffix[len(suffix) - i - 1:]) for i in range(0, len(suffix), 2)]

    # suffixes = [suffixes[-1]]
    # print(len(cex), len(suffixes[-1]))
    return suffixes
