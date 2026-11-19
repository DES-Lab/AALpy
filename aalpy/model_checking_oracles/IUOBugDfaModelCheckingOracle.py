# Model checking oracle that checks for given Mealy machines with input alphabet I and output alphabet O whether their language is rejected by a Dfa with alphabet I union O.
from aalpy.automata.Dfa import Dfa, DfaState
from aalpy.automata.MealyMachine import MealyMachine
from aalpy.base.Oracle import Oracle
from aalpy.utils.HelperFunctions import IUO_dfa_from_IXO_dfa, IXO_dfa_from_mealy

from typing import Any, Callable, List, Tuple


class IUOBugDfaModelCheckingOracle(Oracle):
    """
    Model checking oracle that determines for given Mealy machines with input alphabet I and output
    alphabet O whether their language is rejected by a given property. This property is represented
    by a Dfa which encodes traces over the union of I and O which violate the property.
    """

    def __init__(self,
                 bug_dfa: Dfa,
                 mealy_input_to_dfa_input: Callable[[Any],Any],
                 mealy_output_to_dfa_output: Callable[[Any],Any],
                 is_dfa_input: Callable[[Any],bool],
                 dfa_letter_to_mealy_letter: Callable[[Any],Any],
                 bug_dfa_fallback_letter: Any = None,
                 mealy_to_dfa: Callable[[MealyMachine],Dfa] | None = None):
        """
        Create a model checking oracle which checks Mealy machines against a property represented by
        a Dfa which accepts traces over the union of the Mealy machine's inputs and outputs which
        violate that property. We make the assumption that the Dfa's letters can be recognized as
        either inputs or outputs from the Mealy machines.

        :param Dfa bug_dfa: Dfa that accepts the traces which violate some property.
        :param Callable[[Any],Any] mealy_input_to_dfa_input: Function that maps inputs from the Mealy
            machine to letters from the bug_dfa's alphabet.
        :param Callable[[Any],Any] mealy_output_to_dfa_output: Function that maps outputs from the
            Mealy machine to letters from the bug_dfa's alphabet.
        :param Callable[[Any],bool] is_dfa_input: Predicate function that tells whether a given Dfa
            letter corresponds to an input in the Mealy machines.
        :param Callable[[Any],Any] dfa_letter_to_mealy_letter: Function that maps letters from bug_dfa
            to inputs or outputs from the Mealy machines.
        :param Any bug_dfa_fallback_letter: Optional letter in the bug_dfa's alphabet for which the
            bug_dfa should perform a state transition whenever an input or output from the Mealy
            machine cannot be matched to the letters for any of its other state transitions.
        :param Callable[[MealyMachine],Dfa] | None mealy_to_dfa: Optional alternative for the default
            function used to convert the Mealy machines into Dfas.
        """
        super().__init__(None, None)
        assert isinstance(bug_dfa, Dfa)
        self.bug_dfa = bug_dfa
        self.mealy_input_to_dfa_input = mealy_input_to_dfa_input
        self.mealy_output_to_dfa_output = mealy_output_to_dfa_output
        self.is_dfa_input = is_dfa_input
        self.dfa_letter_to_mealy_letter = dfa_letter_to_mealy_letter
        self.bug_dfa_fallback_letter = bug_dfa_fallback_letter

        self.mealy_to_dfa = self.default_mealy_to_dfa
        if mealy_to_dfa is not None:
            self.mealy_to_dfa = mealy_to_dfa

        self.num_checks_performed = 0
        self.num_counterexamples_found = 0

    def find_cex(self, mealy: MealyMachine) -> list | None:
        """
        Determine for a given Mealy machine whether it satisfies the property encoded by self.bug_dfa.

        :param MealyMachine mealy: Mealy machine.
        :return list | none: If mealy violates the property then an input sequence in mealy that is
            rejected by the property. None if mealy satisfies the property.
        """
        self.num_checks_performed += 1

        assert isinstance(mealy, MealyMachine)
        m_dfa = self.mealy_to_dfa(mealy)

        
        init_states_pair = (m_dfa.initial_state, self.bug_dfa.initial_state)

        queue: List[Tuple[DfaState,DfaState]] = [init_states_pair]
        explored = set([init_states_pair])

        # We don't check whether the pair of initial states is accepting, because this could yield
        # counterexamples that are zero letters long. This would be an issue, since we translate
        # any counterexample we find into a sequence for the given Mealy machine and model checking
        # counterexamples for Mealy machines should always contain at least one input.

        parents = {init_states_pair: None}
        letters_to_parents = {init_states_pair: None}

        while len(queue) > 0:
            current_pair = queue.pop(0)
            (m_state, bug_state) = current_pair

            for letter, m_successor in m_state.transitions.items():

                if letter in bug_state.transitions:
                    bug_successor = bug_state.transitions[letter]
                else:
                    if self.bug_dfa_fallback_letter is None:
                        raise ValueError(f"Bug DFA state `{bug_state.state_id}' doesn't have a transition for the Mealy machine letter `{letter}' and no fallback letter was provided")
                    elif self.bug_dfa_fallback_letter not in bug_state.transitions:
                        raise ValueError(f"Bug DFA state `{bug_state.state_id}' doesn't have a transition for either the Mealy machine letter `{letter}' or the fallback letter `{self.bug_dfa_fallback_letter}'")

                    bug_successor = bug_state.transitions[self.bug_dfa_fallback_letter]

                next_pair = (m_successor, bug_successor)

                if next_pair not in explored:
                    queue.append(next_pair)
                    explored.add(next_pair)

                    parents[next_pair] = current_pair
                    letters_to_parents[next_pair] = letter

                    if bug_successor.is_accepting and m_successor.is_accepting:
                        word = []

                        p = next_pair
                        while p is not None:
                            sym = letters_to_parents[p]
                            if sym is None:
                                break

                            word = [sym] + word
                            p = parents[p]

                        self.num_counterexamples_found += 1
                        return tuple( self.dfa_letter_to_mealy_letter(a) for a in word if self.is_dfa_input(a) )

        return None

    def reset_hyp_and_sul(self, mealy: MealyMachine) -> None:
        """
        Reset a Mealy machine to its initial state.

        :param MealyMachine mealy: Mealy machine.
        """
        mealy.reset_to_initial()
        self.num_queries += 1

    def default_mealy_to_dfa(self, mealy: MealyMachine) -> Dfa:
        """
        Provide a default approach for converting the Mealy machine into a Dfa of which each transition
        is labelled by either an input or output of the Mealy machine. The produced Dfa represents a
        transition q1 -i/o-> q2 from the Mealy machine as q1 -i-> q1_i -o-> q2.

        :param MealyMachine mealy: Mealy machine.
        :return Dfa: Dfa representation of mealy.
        """
        ixo_dfa = IXO_dfa_from_mealy(mealy)
        return IUO_dfa_from_IXO_dfa(ixo_dfa,
                                    self.mealy_input_to_dfa_input,
                                    self.mealy_output_to_dfa_output,
                                    ["bug_state"],
                                    make_input_complete=False,
                                    make_new_states_accepting=True)