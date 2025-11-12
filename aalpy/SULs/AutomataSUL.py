from aalpy.base import Automaton
from aalpy.base import SUL


class AutomatonSUL(SUL):
    def __init__(self, automaton: Automaton):
        super().__init__()
        self.automaton: Automaton = automaton

    def pre(self):
        print('pre')
        self.automaton.reset_to_initial()

    def step(self, letter=None):
        return self.automaton.step(letter)

    def post(self):
        print('post')
        pass


MealySUL = OnfsmSUL = StochasticMealySUL = DfaSUL = MooreSUL = MdpSUL = McSUL = SevpaSUL = AutomatonSUL
