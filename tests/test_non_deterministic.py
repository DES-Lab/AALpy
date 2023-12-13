import unittest


class NonDeterministicTest(unittest.TestCase):

    def test_non_det(self):

        from aalpy.SULs import AutomatonSUL
        from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle
        from aalpy.learning_algs import run_non_det_Lstar
        from aalpy.utils import get_benchmark_ONFSM

        onfsm = get_benchmark_ONFSM()
        alphabet = onfsm.get_input_alphabet()

        for _ in range(100):
            sul = AutomatonSUL(onfsm)

            oracle = RandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=2, max_walk_len=5)

            learned_onfsm = run_non_det_Lstar(alphabet, sul, oracle, n_sampling=50, print_level=0)

            eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=10000, reset_prob=0.09,
                                                       reset_after_cex=True)

            cex = eq_oracle.find_cex(learned_onfsm)

            if cex or len(learned_onfsm.states) != len(onfsm.states):
                assert False
        assert True
