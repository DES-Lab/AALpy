from aalpy.base import Oracle, SUL
from aalpy.utils import visualize_automaton


class UserInputEqOracle(Oracle):
    """
    Interactive equivalence oracle. For every counterexample, the current hypothesis will be visualized and the user can
    enter the counterexample step by step.
    The user provides elements of the input alphabet or commands.
    When the element of the input alphabet is entered, the step will be performed in the current hypothesis and output
    will
    be printed.

    Commands offered to the users are:

        print alphabet - prints the input alphabet

        current inputs - inputs entered so far

        cex - returns inputs entered so far as the counterexample

        end - no counterexample exists

        reset - resets the current state of the hypothesis and clears inputs
    """
    def __init__(self, alphabet: list, sul: SUL):
        super().__init__(alphabet, sul)
        self.curr_hypothesis = 0

    def find_cex(self, hypothesis):

        self.curr_hypothesis += 1
        inputs = []
        visualize_automaton(hypothesis, path=f'Hypothesis_{self.curr_hypothesis}')
        while True:
            inp = input('Please provide an input: ')
            if inp == 'print alphabet':
                print(self.alphabet)
                continue
            if inp == 'current inputs':
                print(inputs)
                continue
            if inp == 'cex':
                if inputs:
                    return inputs
            if inp == 'end':
                return None
            if inp == 'reset':
                inputs.clear()
                hypothesis.reset_to_initial()
                print('You are back in the initial state. Please provide an input: ')
                continue
            if inp not in self.alphabet:
                print("Provided input is not in the input alphabet.")
                continue
            inputs.append(inp)
            out = hypothesis.step(inp)
            print('Output:', out)
