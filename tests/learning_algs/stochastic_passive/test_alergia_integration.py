import os
import random
import tempfile
import unittest
from unittest.mock import patch

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Mdp, MdpState, MarkovChain, McState
from aalpy.learning_algs import run_Alergia, run_active_Alergia
from aalpy.learning_algs.stochastic_passive.ActiveAleriga import RandomWordSampler
from aalpy.learning_algs.stochastic_passive.Alergia import run_JAlergia


def _ground_truth_mdp():
    """
    3-state MDP: s0(A) --a--> s1(B) w.p. 0.7 / s2(C) w.p. 0.3; s0 --b--> s0.
    s1(B)/s2(C) absorb back to s0 on 'a', self-loop on 'b'.
    """
    s0 = MdpState('s0', output='A')
    s1 = MdpState('s1', output='B')
    s2 = MdpState('s2', output='C')
    s0.transitions['a'].append((s1, 0.7))
    s0.transitions['a'].append((s2, 0.3))
    s0.transitions['b'].append((s0, 1.0))
    s1.transitions['a'].append((s0, 1.0))
    s1.transitions['b'].append((s1, 1.0))
    s2.transitions['a'].append((s0, 1.0))
    s2.transitions['b'].append((s2, 1.0))
    return Mdp(s0, [s0, s1, s2])


def _ground_truth_mc():
    """2-state biased coin Markov chain: heads (0.7) self-loop-ish, tails (0.3)."""
    heads = McState('heads', output='H')
    tails = McState('tails', output='T')
    heads.transitions.append((heads, 0.7))
    heads.transitions.append((tails, 0.3))
    tails.transitions.append((heads, 0.6))
    tails.transitions.append((tails, 0.4))
    return MarkovChain(heads, [heads, tails])


def _generate_mdp_data(ground_truth, alphabet, num_traces, trace_len):
    sul = AutomatonSUL(ground_truth)
    data = []
    for _ in range(num_traces):
        walk = [random.choice(alphabet) for _ in range(trace_len)]
        outputs = sul.query(tuple(walk))
        trace = [ground_truth.initial_state.output]
        for inp, out in zip(walk, outputs):
            trace.append((inp, out))
        data.append(trace)
    return data


def _generate_mc_data(ground_truth, num_traces, trace_len):
    data = []
    for _ in range(num_traces):
        ground_truth.reset_to_initial()
        trace = [ground_truth.current_state.output]
        for _ in range(trace_len):
            trace.append(ground_truth.step())
        data.append(trace)
    return data


def _output_distribution(model, input_seq, n_samples):
    counts = {}
    for _ in range(n_samples):
        outputs = tuple(model.execute_sequence(model.initial_state, list(input_seq)))
        counts[outputs] = counts.get(outputs, 0) + 1
    return {k: v / n_samples for k, v in counts.items()}


def _total_variation_distance(dist1, dist2):
    keys = set(dist1) | set(dist2)
    return 0.5 * sum(abs(dist1.get(k, 0) - dist2.get(k, 0)) for k in keys)


class RunAlergiaMdpTest(unittest.TestCase):

    def test_learned_mdp_resembles_ground_truth_output_distributions(self):
        random.seed(10)
        ground_truth = _ground_truth_mdp()
        alphabet = ground_truth.get_input_alphabet()
        data = _generate_mdp_data(ground_truth, alphabet, num_traces=3000, trace_len=4)

        learned_model = run_Alergia(data, automaton_type='mdp', eps=0.3, print_info=False)

        random.seed(99)
        for input_seq in [('a',), ('a', 'a'), ('b', 'a')]:
            truth_dist = _output_distribution(ground_truth, input_seq, 300)
            learned_dist = _output_distribution(learned_model, input_seq, 300)
            tvd = _total_variation_distance(truth_dist, learned_dist)
            self.assertLessEqual(tvd, 0.35, f'{input_seq}: truth={truth_dist} learned={learned_dist}')


class RunAlergiaMcTest(unittest.TestCase):

    def test_learned_markov_chain_resembles_ground_truth_stationary_behaviour(self):
        random.seed(11)
        ground_truth = _ground_truth_mc()
        data = _generate_mc_data(ground_truth, num_traces=3000, trace_len=5)

        learned_model = run_Alergia(data, automaton_type='mc', eps=0.3, print_info=False)

        def first_step_distribution(model, n):
            counts = {'H': 0, 'T': 0}
            for _ in range(n):
                model.reset_to_initial()
                out = model.step()
                counts[out] = counts.get(out, 0) + 1
            return {k: v / n for k, v in counts.items()}

        random.seed(55)
        truth_dist = first_step_distribution(ground_truth, 1000)
        learned_dist = first_step_distribution(learned_model, 1000)
        tvd = _total_variation_distance(truth_dist, learned_dist)
        self.assertLessEqual(tvd, 0.2, f'truth={truth_dist} learned={learned_dist}')


class RunJAlergiaTest(unittest.TestCase):

    def test_missing_jar_returns_none_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_jar = os.path.join(tmp_dir, 'does_not_exist.jar')
            data_file = os.path.join(tmp_dir, 'data.txt')
            with open(data_file, 'w') as f:
                f.write('A,a,B\n')

            result = run_JAlergia(data_file, 'mdp', missing_jar)

            self.assertIsNone(result)


class RandomWordSamplerTest(unittest.TestCase):

    def test_sample_uses_true_initial_output_and_aligned_input_output_pairs(self):
        # Regression test: RandomWordSampler.sample() used to treat the output observed after the *first*
        # input as if it were the SUL's initial output, and then paired random_walk[i] with the output
        # observed after input i+1 (an off-by-one misalignment). Both bugs corrupted every generated sample
        # and could even make consistent MDP data collection impossible (differing "initial outputs" trip
        # the FPTA's initial-output consistency assertion).
        random.seed(30)
        # deterministic (prob-1.0 only) ground truth so re-querying the same inputs later is reproducible
        s0 = MdpState('s0', output='A')
        s1 = MdpState('s1', output='B')
        s0.transitions['a'].append((s1, 1.0))
        s0.transitions['b'].append((s0, 1.0))
        s1.transitions['a'].append((s0, 1.0))
        s1.transitions['b'].append((s1, 1.0))
        ground_truth = Mdp(s0, [s0, s1])
        sul = AutomatonSUL(ground_truth)
        sampler = RandomWordSampler(num_walks=20, min_walk_len=3, max_walk_len=3)

        samples = sampler.sample(sul, ground_truth)

        for sample in samples:
            initial_output, *steps = sample
            self.assertEqual(initial_output, ground_truth.initial_state.output)

            inputs = tuple(io[0] for io in steps)
            outputs = tuple(io[1] for io in steps)
            expected_outputs = tuple(sul.query(inputs))
            self.assertEqual(outputs, expected_outputs)


class RunActiveAlergiaTest(unittest.TestCase):

    def test_active_alergia_improves_model_over_iterations(self):
        random.seed(21)
        ground_truth = _ground_truth_mdp()
        alphabet = ground_truth.get_input_alphabet()
        initial_data = _generate_mdp_data(ground_truth, alphabet, num_traces=200, trace_len=3)

        sul = AutomatonSUL(ground_truth)
        sampler = RandomWordSampler(num_walks=200, min_walk_len=2, max_walk_len=4)

        learned_model = run_active_Alergia(initial_data, sul, sampler, n_iter=3, eps=0.3, print_info=False)

        random.seed(88)
        for input_seq in [('a',), ('b', 'a')]:
            truth_dist = _output_distribution(ground_truth, input_seq, 300)
            learned_dist = _output_distribution(learned_model, input_seq, 300)
            tvd = _total_variation_distance(truth_dist, learned_dist)
            self.assertLessEqual(tvd, 0.35, f'{input_seq}: truth={truth_dist} learned={learned_dist}')

    def test_active_alergia_forwards_automaton_type_to_run_alergia(self):
        # Regression test: run_active_Alergia used to hardcode automaton_type='mdp' when calling run_Alergia,
        # silently ignoring its own automaton_type parameter. Verify the parameter is now actually forwarded.
        data = [['A', ('a', 'B')]]
        sul = object()

        class _NoopSampler:
            def sample(self, sul, model):
                return []

        with patch('aalpy.learning_algs.stochastic_passive.ActiveAleriga.run_Alergia') as mock_run_alergia:
            mock_run_alergia.return_value = 'fake-model'
            run_active_Alergia(data, sul, _NoopSampler(), n_iter=1, automaton_type='smm', print_info=False)

        self.assertEqual(mock_run_alergia.call_args.kwargs['automaton_type'], 'smm')


if __name__ == '__main__':
    unittest.main()
