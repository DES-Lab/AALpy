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

            cex = prop.find_cex(hypothesis)

            # If the property oracle rejects the hypothesis for cex
            if cex is not None:
                cex = tuple(cex)

                hyp_out = tuple(hypothesis.execute_sequence(hypothesis.initial_state, cex))
                sul_out = tuple(self.sul.query(cex))

                # If the SUL and the hypothesis have the same outputs for cex, then the SUL violates prop as well
                if hyp_out == sul_out:
                    self.violated_properties.add(label)

                    if self.property_violation_callback is not None:
                        self.property_violation_callback(label, cex)
                elif prop_cex is None:
                    # The hypothesis and the SUL have distinct outputs for cex
                    prop_cex = cex

                    # Take the minimal distinguishing prefix of the counterexample
                    for i in range(min(len(hyp_out), len(sul_out))):
                        if hyp_out[i] != sul_out[i]:
                            prop_cex = prop_cex[:i + 1]
                            break

                    if not self.check_all_props_when_a_first_prop_cex_was_found:
                        return cex

        # Return the property counterexample, if one was found
        if prop_cex is not None:
            return prop_cex

        # Perform equivalence checking
        cex = self.eq_oracle.find_cex(hypothesis)
        if cex is not None:
            cex = tuple(cex)

        return cex