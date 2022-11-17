from aalpy.base import SUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.learning_algs.deterministic.LStar import run_Lstar


class ThreePlaceBufferExample(SUL):
    
    def __init__(self):
        super().__init__()
        self.buf = []
        self.alphabet = ["get", "put"]

    def pre(self):
        self.buf = []

    def put(self):
        if (len(self.buf) < 3):
            self.buf.append("e")
            return "OK"
        return "X"

    def get(self):
        if (len(self.buf) > 0):
            self.buf.pop()
            return "OK"
        return "X"

    def post(self):
        pass

    def step(self, letter):
        if letter == "get":
            return self.get()
        return self.put()


def main():
    sul = ThreePlaceBufferExample()
    eq_oracle = RandomWalkEqOracle(sul.alphabet, sul, 1000)

    learned_dfa = run_KV(sul.alphabet, sul, eq_oracle, automaton_type='mealy', print_level=1, cex_processing=None)

    eq_oracle = RandomWalkEqOracle(sul.alphabet, sul, 1000)
    #learned_dfa = run_Lstar(sul.alphabet, sul, eq_oracle, automaton_type='mealy', print_level=3, cex_processing="rs")

    #learned_dfa.visualize()


if __name__ == "__main__":
    main()
