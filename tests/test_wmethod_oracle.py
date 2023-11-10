import unittest

try:
    from aalpy.automata import MooreMachine, MooreState
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles.WMethodEqOracle import WMethodEqOracle
    from aalpy.SULs import AutomatonSUL
    from aalpy.utils import visualize_automaton
except ImportError:
    import sys
    from pathlib import Path

    # if you want to run the test directly from CLI
    # either from root or from tests folder
    p = Path(__file__).parent.resolve()
    sys.path.append(str(p))
    sys.path.append(str(p.parent))
    from aalpy.automata import MooreMachine, MooreState
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles.WMethodEqOracle import WMethodEqOracle
    from aalpy.SULs import AutomatonSUL
    from aalpy.utils import visualize_automaton


class TestWMethodOracle(unittest.TestCase):
    @staticmethod
    def gen_moore_from_state_setup(state_setup) -> MooreMachine:
        # state_setup shoud map from state_id to tuple(output and transitions_dict)

        # build states with state_id and output
        states = {key: MooreState(key, val[0]) for key, val in state_setup.items()}

        # add transitions to states
        for state_id, state in states.items():
            for _input, target_state_id in state_setup[state_id][1].items():
                state.transitions[_input] = states[target_state_id]

        # states to list
        states = [state for state in states.values()]

        # build moore machine with first state as starting state
        mm = MooreMachine(states[0], states)

        for state in states:
            state.prefix = mm.get_shortest_path(mm.initial_state, state)

        return mm

    def generate_real_automata(self) -> MooreMachine:
        state_setup = {
            "a": ("a", {"x": "b1", "y": "a"}),
            "b1": ("b", {"x": "b2", "y": "a"}),
            "b2": ("b", {"x": "b3", "y": "a"}),
            "b3": ("b", {"x": "b4", "y": "a"}),
            "b4": ("b", {"x": "c", "y": "a"}),
            "c": ("c", {"x": "a", "y": "a"}),
        }

        mm = self.gen_moore_from_state_setup(state_setup)
        mm.characterization_set = mm.compute_characterization_set() + [tuple()]
        return mm

    def generate_hypothesis(self) -> MooreMachine:
        state_setup = {
            "a": ("a", {"x": "b", "y": "a"}),
            "b": ("b", {"x": "b", "y": "a"}),
        }

        mm = self.gen_moore_from_state_setup(state_setup)
        # ! computer_characterization_set does not work for Moore machines in general!
        # mm.characterization_set = mm.compute_characterization_set() + [tuple()]
        mm.characterization_set = [tuple(), ("x",), ("y",)]
        return mm

    def test_wmethod_oracle(self):
        real = self.generate_real_automata()
        hyp = self.generate_hypothesis()
        # visualize_automaton(real)
        # visualize_automaton(hyp)
        assert set(real.get_input_alphabet()) == {"x", "y"}
        assert set(hyp.get_input_alphabet()) == {"x", "y"}
        assert len(real.states) == 6
        assert len(hyp.states) == 2
        alphabet = real.get_input_alphabet()
        oracle = WMethodEqOracle(
            alphabet, AutomatonSUL(real), len(real.states) + 1, shuffle_test_set=False
        )
        cex = oracle.find_cex(hyp)
        assert cex is not None, "Expected a counterexample, but got None"

    def test_wmethod_oracle_with_lstar(self):
        real = self.generate_real_automata()
        hyp = self.generate_hypothesis()
        # visualize_automaton(real)
        # visualize_automaton(hyp)
        assert real.get_input_alphabet() == ["x", "y"]
        assert hyp.get_input_alphabet() == ["x", "y"]
        assert len(real.states) == 6
        assert len(hyp.states) == 2
        alphabet = real.get_input_alphabet()
        oracle = WMethodEqOracle(
            alphabet, AutomatonSUL(real), len(real.states) + 1, shuffle_test_set=False
        )
        lstar_hyp = run_Lstar(alphabet, AutomatonSUL(real), oracle, "moore")
        # print(lstar_hyp)
        # visualize_automaton(lstar_hyp)
        assert (
            len(lstar_hyp.states) == 6
        ), f"Expected {6} states got {len(lstar_hyp.states)} in lstar hypothesis"


if __name__ == "__main__":
    unittest.main()
