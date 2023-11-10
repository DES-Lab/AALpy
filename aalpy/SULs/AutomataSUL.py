from aalpy.base import SUL, Automaton
from aalpy.automata import MarkovChain, Mdp


class AutomatonSUL(SUL):
    def __init__(self, automaton: Automaton):
        super().__init__()
        self.automaton: Automaton = automaton

    def pre(self):
        self.automaton.reset_to_initial()
        if isinstance(self.automaton, (MarkovChain, Mdp)):
            return self.automaton.initial_state.output

    def step(self, letter=None):
        return self.automaton.step(letter)

    def post(self):
        pass

    def query(self, word: tuple) -> list:
        output = super().query(word)
        if isinstance(self.automaton, (MarkovChain, Mdp)):
            output.insert(0, self.automaton.initial_state.output)
        return output


MealySUL = OnfsmSUL = StochasticMealySUL = DfaSUL = MooreSUL = MdpSUL = McSUL = AutomatonSUL
