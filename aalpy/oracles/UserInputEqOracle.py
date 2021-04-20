from aalpy.base import Oracle, SUL
from aalpy.utils import visualize_automaton


class UserInputEqOracle(Oracle):
    """
    Interactive equivalence oracle. For every counterexample, the current hypothesis will be visualized and the user can
    enter the counterexample step by step.
    The user provides elements of the input alphabet or commands.
    When the element of the input alphabet is entered, the step will be performed in the current hypothesis and output
    will be printed.

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

        self.reset_hyp_and_sul(hypothesis)

        self.curr_hypothesis += 1
        inputs = []
        visualize_automaton(hypothesis, path=f'Hypothesis_{self.curr_hypothesis}')
        while True:
            inp = input('Please provide an input: ')
            if inp == 'help':
                print('Use one of following commands [print alphabet, current inputs, cex, end, reset] '
                      'or provide an input')
                continue
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
                self.reset_hyp_and_sul(hypothesis)
                print('You are back in the initial state. Please provide an input: ')
                continue
            if inp not in self.alphabet:
                print("Provided input is not in the input alphabet.")
                continue
            inputs.append(inp)
            self.num_steps += 1
            out_hyp = hypothesis.step(inp)
            out_sul = self.sul.step(inp)
            print('Hypothesis Output :', out_hyp)
            print('SUL Output        :', out_sul)
            if out_hyp != out_sul:
                print('Counterexample found.\nIf you want to return it, type \'end\'.')
