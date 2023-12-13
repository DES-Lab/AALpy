from aalpy.base import SUL, Automaton
from aalpy.automata import MarkovChain, Mdp
from aalpy.base import SUL
from aalpy.automata import Dfa, MealyMachine, MooreMachine, Onfsm, Mdp, StochasticMealyMachine, MarkovChain, Sevpa


class AutomatonSUL(SUL):
    def __init__(self, automaton: Automaton):
        super().__init__()
        self.automaton: Automaton = automaton
        self.include_initial_output = isinstance(self.automaton, (MarkovChain, Mdp))

    def pre(self):
        self.automaton.reset_to_initial()
        if self.include_initial_output:
            return self.automaton.initial_state.output

    def step(self, letter=None):
        return self.automaton.step(letter)

    def post(self):
        pass

    def query(self, word: tuple) -> list:
        initial_output = self.pre()
        out = [self.step(i) for i in word]
        if initial_output:
            out.insert(0, initial_output)
        self.post()
        return out


MealySUL = OnfsmSUL = StochasticMealySUL = DfaSUL = MooreSUL = MdpSUL = McSUL = AutomatonSUL


class SevpaSUL(SUL):
    def __init__(self, sevpa: Sevpa):
        super().__init__()
        self.sevpa = sevpa

    def pre(self):
        self.sevpa.reset_to_initial()

    def post(self):
        pass

    def step(self, letter):
        return self.sevpa.step(letter)
