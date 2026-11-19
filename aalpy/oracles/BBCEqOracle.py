# Equivalence oracle that performs black-box checking for a given collection of properties.
from aalpy.base import Automaton, Oracle, SUL
from typing import Callable, Dict


class BBCEqOracle(Oracle):
    """Equivalence oracle wrapper that performs black-box checking (BBC) for oracles that represent
    the properties against which the hypotheses should be checked.
    """

    def __init__(self,
                 eq_oracle: Oracle,
                 property_oracles: Dict[str,Oracle],
                 property_violation_callback: Callable[[str,tuple],None] | None = None,
                 check_all_props_when_a_first_prop_cex_was_found: bool = True):
        """
        Create a BBC equivalence oracle.

        :param Oracle eq_oracle: Wrapped equivalence oracle.
        :param Dict[str,Oracle] property_oracles: Dictionary that maps string-typed labels to the Oracles
            that represent the properties against which the hypotheses should be checked.
        :param Callable[[str,tuple],None] | None property_violation_callback: Optional callback to be
            called when a monitor is found to be violated by the SUL. The monitor's label and the
            counterexample input sequence are then passed to the callback as arguments.
        :param bool check_all_props_when_a_first_prop_cex_was_found: Check all properties against the
            hypothesis even if a counterexample to be used for hypothesis refinement by the learner is
            found before all of them are checked. 
        """
        self.eq_oracle = eq_oracle

        self.property_oracles = property_oracles
        self.property_violation_callback = property_violation_callback
        self.check_all_props_when_a_first_prop_cex_was_found = check_all_props_when_a_first_prop_cex_was_found

        # Keep track of the properties for which counterexamples have been found
        self.violated_properties = set()


    # Bind the alphabet to that of the wrapped equivalence oracle.
    @property
    def alphabet(self) -> list:
        return self.eq_oracle.alphabet
    @alphabet.setter
    def alphabet(self, value: list) -> None:
        self.eq_oracle.alphabet = value

    # Bind the system under learning to that of the wrapped equivalence oracle.
    @property
    def sul(self) -> SUL:
        return self.eq_oracle.sul
    @sul.setter
    def sul(self, value: SUL) -> None:
        self.eq_oracle.sul = value

    # Bind the number of queries to that of the wrapped equivalence oracle.
    @property
    def num_queries(self) -> int:
        return self.eq_oracle.num_queries
    @num_queries.setter
    def num_queries(self, value: int) -> None:
        self.eq_oracle.num_queries = value

    # Bind the number of steps to that of the wrapped equivalence oracle.
    @property
    def num_steps(self) -> int:
        return self.eq_oracle.num_steps
    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self.eq_oracle.num_steps = value


    def forget_all_property_violations(self) -> None:
        """
        Forget for which properties counterexamples have been confirmed against the SUL, so that all
        properties will be checked again in the next call of self.find_cex.
        """
        self.violated_properties = set()

    def find_cex(self, hypothesis: Automaton) -> tuple | None:
        """
        Return a counterexample (inputs) that displays different behavior on system under learning and
        current hypothesis. The hypothesis is initially checked against each of the properties for which
        no counterexamples have been confirmed against the SUL.

        :param Automaton hypothesis: Current hypothesis.
        :return Tuple[InputType, ...] | None: Counterexample inputs, None if no counterexample is found.
        """
        if len(hypothesis.states) == 0:
            return None

        prop_cex = None
        for label, prop in self.property_oracles.items():
            # Skip this property if a SUL counterexample was already found
            if label in self.violated_properties:
                continue

            # Check the hypothesis against prop
            cex = prop.find_cex(hypothesis)

            # If the property oracle rejects the hypothesis for cex
            if cex is not None:
                cex = tuple(cex)

                # Check cex against the SUL
                self.reset_hyp_and_sul(hypothesis)
                cex_rejected_by_sul = False
                for ind, letter in enumerate(cex):
                    out_h = hypothesis.step(letter)
                    out_s = self.sul.step(letter)
                    self.num_steps += 1

                    if out_h != out_s:
                        # Cex is not a valid counterexample for the SUL
                        cex_rejected_by_sul = True

                        # Remember the relevant prefix for cex, as it can be used for hypothesis refinement
                        if prop_cex is None:
                            prop_cex = tuple(cex[:ind + 1])

                            if not self.check_all_props_when_a_first_prop_cex_was_found:
                                self.sul.post()
                                return prop_cex

                        # The SUL rejected cex
                        break

                # Clean up the SUL after running cex
                self.sul.post()

                # If the SUL and the hypothesis have the same outputs for cex, then the SUL violates prop as well
                if not cex_rejected_by_sul:
                    self.violated_properties.add(label)

                    if self.property_violation_callback is not None:
                        self.property_violation_callback(label, cex)

        # Return the property counterexample, if one was found. Prop_cex is only set if the hypothesis and the SUL
        # have distinct outputs for it, so it can be used for hypothesis refinement
        if prop_cex is not None:
            return prop_cex

        # Perform equivalence checking
        cex = self.eq_oracle.find_cex(hypothesis)
        if cex is not None:
            cex = tuple(cex)

        return cex