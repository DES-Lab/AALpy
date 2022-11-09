from aalpy.base import SUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.learning_algs.deterministic.LStar import run_Lstar


class KVBookExample(SUL):
    
    def __init__(self):
        super().__init__()
        self.string = ""
        self.alphabet = ["0", "1"]

    def pre(self):
        self.string = ""

    def post(self):
        pass

    def step(self, letter):
        if letter != None:
            self.string = self.string + letter
        return self.string.count("1") % 4 == 3


def main():
    sul = KVBookExample()
    eq_oracle = RandomWalkEqOracle(sul.alphabet, sul, 500)

    learned_dfa = run_KV(sul.alphabet, sul, eq_oracle, automaton_type='dfa',
                        print_level=3, reuse_counterexamples=True, cex_processing="rs")

    learned_dfa = run_Lstar(sul.alphabet, sul, eq_oracle, automaton_type='dfa',
                        print_level=3, cex_processing="rs")

    #    learned_dfa = run_Lstar(sul.alphabet, sul, eq_oracle, automaton_type="dfa")


    learned_dfa.visualize()




if __name__ == "__main__":
    main()
