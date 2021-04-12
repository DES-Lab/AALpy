
def random_mealy_example(alphabet_size, number_of_states, output_size=8):
    """
    Generate a random Mealy machine and learn it.
    :param alphabet_size: size of input alphabet
    :param number_of_states: number of states in generated Mealy machine
    :param output_size: size of the output
    :return: learned Mealy machine
    """

    from aalpy.SULs import MealySUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import RandomWalkEqOracle, StatePrefixEqOracle

    alphabet = [*range(0, alphabet_size)]

    from aalpy.utils import generate_random_mealy_machine
    random_mealy = generate_random_mealy_machine(number_of_states, alphabet, output_alphabet=list(range(output_size)))

    sul_mealy = MealySUL(random_mealy)

    random_walk_eq_oracle = RandomWalkEqOracle(alphabet, sul_mealy, 5000)
    state_origin_eq_oracle = StatePrefixEqOracle(alphabet, sul_mealy, walks_per_state=10, walk_len=15)

    learned_mealy = run_Lstar(alphabet, sul_mealy, random_walk_eq_oracle, automaton_type='mealy',
                              cex_processing='longest_prefix')

    return learned_mealy


def random_moore_example(alphabet_size, number_of_states, output_size=8):
    """
    Generate a random Moore machine and learn it.
    :param alphabet_size: size of input alphabet
    :param number_of_states: number of states in generated Mealy machine
    :param output_size: size of the output
    :return: learned Moore machine
    """
    alphabet = [*range(0, alphabet_size)]

    from aalpy.SULs import MooreSUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import StatePrefixEqOracle
    from aalpy.utils import generate_random_moore_machine

    random_moore = generate_random_moore_machine(number_of_states, alphabet, output_alphabet=list(range(output_size)))

    sul_mealy = MooreSUL(random_moore)

    state_origin_eq_oracle = StatePrefixEqOracle(alphabet, sul_mealy, walks_per_state=15, walk_len=20)
    learned_moore = run_Lstar(alphabet, sul_mealy, state_origin_eq_oracle, cex_processing='rs',
                              closing_strategy='single', automaton_type='moore', cache_and_non_det_check=True)
    return learned_moore


def random_dfa_example(alphabet_size, number_of_states, num_accepting_states=1):
    """
    Generate a random DFA machine and learn it.
    :param alphabet_size: size of the input alphabet
    :param number_of_states: number of states in the generated DFA
    :param num_accepting_states: number of accepting states
    :return: DFA
    """
    import string
    from aalpy.SULs import DfaSUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import StatePrefixEqOracle, TransitionFocusOracle, WMethodEqOracle, \
        RandomWalkEqOracle, RandomWMethodEqOracle, BreadthFirstExplorationEqOracle, RandomWordEqOracle,\
        CacheBasedEqOracle,  UserInputEqOracle, KWayStateCoverageEqOracle
    from aalpy.utils import generate_random_dfa

    assert num_accepting_states <= number_of_states

    alphabet = list(string.ascii_letters[:26])[:alphabet_size]
    random_dfa = generate_random_dfa(number_of_states, alphabet, num_accepting_states)
    alphabet = list(string.ascii_letters[:26])[:alphabet_size]
    # visualize_automaton(random_dfa, path='correct')
    sul_dfa = DfaSUL(random_dfa)

    # examples of various equivalence oracles

    random_walk_eq_oracle = RandomWalkEqOracle(alphabet, sul_dfa, 5000)
    state_origin_eq_oracle = StatePrefixEqOracle(alphabet, sul_dfa, walks_per_state=10, walk_len=50)
    tran_cov_eq_oracle = TransitionFocusOracle(alphabet, sul_dfa, num_random_walks=200, walk_len=30,
                                               same_state_prob=0.3)
    w_method_eq_oracle = WMethodEqOracle(alphabet, sul_dfa, max_number_of_states=number_of_states)
    random_W_method_eq_oracle = RandomWMethodEqOracle(alphabet, sul_dfa, walks_per_state=10, walk_len=50)
    bf_exploration_eq_oracle = BreadthFirstExplorationEqOracle(alphabet, sul_dfa, 5)
    random_word_eq_oracle = RandomWordEqOracle(alphabet, sul_dfa)
    cache_based_eq_oracle = CacheBasedEqOracle(alphabet, sul_dfa)
    user_based_eq_oracle = UserInputEqOracle(alphabet, sul_dfa)
    kWayStateCoverageEqOracle = KWayStateCoverageEqOracle(alphabet, sul_dfa)
    learned_dfa = run_Lstar(alphabet, sul_dfa, random_walk_eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=False, cex_processing='rs')

    # visualize_automaton(learned_dfa)
    return learned_dfa


def random_onfsm_example(num_states, input_size, output_size, n_sampling):
    """
    Generate and learn random ONFSM.
    :param num_states: number of states of the randomly generated automaton
    :param input_size: size of the input alphabet
    :param output_size: size of the output alphabet
    :param n_sampling: number of times each query will be repeated to ensure that all non-determinist outputs are
    observed
    :return: learned ONFSM
    """
    from aalpy.SULs import OnfsmSUL
    from aalpy.utils import generate_random_ONFSM
    from aalpy.oracles import UnseenOutputRandomWalkEqOracle, UnseenOutputRandomWordEqOracle
    from aalpy.learning_algs import run_Lstar_ONFSM

    onfsm = generate_random_ONFSM(num_states=num_states, num_inputs=input_size, num_outputs=output_size)
    alphabet = onfsm.get_input_alphabet()

    sul = OnfsmSUL(onfsm)
    eq_oracle = UnseenOutputRandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=10, max_walk_len=50)
    eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.15, reset_after_cex=True)

    learned_model = run_Lstar_ONFSM(alphabet, sul, eq_oracle=eq_oracle, n_sampling=n_sampling)
    return learned_model


def random_mdp_example(num_states, input_len, num_outputs, n_c=20, n_resample=1000, min_rounds=10, max_rounds=1000):
    """
    Generate and learn random MDP.
    :param num_states: number of states in generated MDP
    :param input_len: size of input alphabet
    :param n_c: cutoff for a state to be considered complete
    :param n_resample: resampling size
    :param num_outputs: size of output alphabet
    :param min_rounds: minimum number of learning rounds
    :param max_rounds: maximum number of learning rounds
    :return: learned MDP
    """
    from aalpy.SULs import MdpSUL
    from aalpy.oracles import UnseenOutputRandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import generate_random_mdp

    mdp, input_alphabet = generate_random_mdp(num_states, input_len, num_outputs)
    sul = MdpSUL(mdp)
    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=5000, reset_prob=0.11,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=n_c, n_resample=n_resample,
                                       min_rounds=min_rounds, max_rounds=max_rounds)

    return learned_mdp


def angluin_seminal_example():
    """
    Example automaton from Angluin's seminal paper.
    :return: learned DFA
    """
    from aalpy.SULs import DfaSUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import get_Angluin_dfa

    dfa = get_Angluin_dfa()

    alphabet = dfa.get_input_alphabet()

    sul = DfaSUL(dfa)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500)

    learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=True, cex_processing=None, print_level=3)

    return learned_dfa


def tomita_example(tomita_number):
    """
    Pass a tomita function to this example and learn it.
    :param: function of the desired tomita grammar
    :rtype: Dfa
    :return DFA representing tomita grammar
    """
    from aalpy.SULs import TomitaSUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import StatePrefixEqOracle

    tomita_sul = TomitaSUL(tomita_number)
    alphabet = [0, 1]
    state_origin_eq_oracle = StatePrefixEqOracle(alphabet, tomita_sul, walks_per_state=20, walk_len=10)

    learned_dfa = run_Lstar(alphabet, tomita_sul, state_origin_eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=True)

    return learned_dfa


def regex_example(regex, alphabet):
    """
    Learn a regular expression.
    :param regex: regex to learn
    :param alphabet: alphabet of the regex
    :return: DFA representing the regex
    """
    from aalpy.SULs import RegexSUL
    from aalpy.oracles import StatePrefixEqOracle
    from aalpy.learning_algs import run_Lstar

    regex_sul = RegexSUL(regex)

    eq_oracle = StatePrefixEqOracle(alphabet, regex_sul, walks_per_state=20,
                                    walk_len=10)

    learned_regex = run_Lstar(alphabet, regex_sul, eq_oracle, automaton_type='dfa')

    return learned_regex


def learn_python_class():
    """
    Learn a Mealy machine where inputs are methods and arguments of the class that serves as SUL.
    :return: Mealy machine
    """

    # class
    from aalpy.SULs import PyClassSUL, FunctionDecorator
    from aalpy.oracles import StatePrefixEqOracle
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import MockMqttExample, visualize_automaton

    mqtt = MockMqttExample

    input_al = [FunctionDecorator(mqtt.connect), FunctionDecorator(mqtt.disconnect),
                FunctionDecorator(mqtt.subscribe, 'topic'), FunctionDecorator(mqtt.unsubscribe, 'topic'),
                FunctionDecorator(mqtt.publish, 'topic')]

    sul = PyClassSUL(mqtt)

    eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=20, walk_len=20)

    mealy = run_Lstar(input_al, sul, eq_oracle=eq_oracle, automaton_type='mealy', cache_and_non_det_check=True)

    visualize_automaton(mealy)


def mqtt_example():
    from aalpy.base import SUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import visualize_automaton, MockMqttExample

    class MQTT_SUL(SUL):
        def __init__(self):
            super().__init__()
            self.mqtt = MockMqttExample()

        def pre(self):
            self.mqtt.state = 'CONCLOSED'

        def post(self):
            self.mqtt.topics.clear()

        def step(self, letter):
            if letter == 'connect':
                return self.mqtt.connect()
            elif letter == 'disconnect':
                return self.mqtt.disconnect()
            elif letter == 'publish':
                return self.mqtt.publish(topic='test')
            elif letter == 'subscribe':
                return self.mqtt.subscribe(topic='test')
            else:
                return self.mqtt.unsubscribe(topic='test')

    sul = MQTT_SUL()
    input_al = ['connect', 'disconnect', 'publish', 'subscribe', 'unsubscribe']

    eq_oracle = RandomWalkEqOracle(input_al, sul, num_steps=2000, reset_after_cex=True, reset_prob=0.15)

    mealy = run_Lstar(input_al, sul, eq_oracle=eq_oracle, automaton_type='mealy', cache_and_non_det_check=True,
                      print_level=3)

    visualize_automaton(mealy)


def onfsm_mealy_paper_example():
    """
    Learning a ONFSM presented in 'Learning Finite State Models of Observable Nondeterministic Systems in a Testing
    Context'.
    :return: learned ONFSM
    """

    from random import seed
    from aalpy.SULs import OnfsmSUL
    from aalpy.oracles import UnseenOutputRandomWordEqOracle, UnseenOutputRandomWalkEqOracle
    from aalpy.learning_algs import run_Lstar_ONFSM
    from aalpy.utils import get_benchmark_ONFSM
    seed(3)

    onfsm = get_benchmark_ONFSM()
    alphabet = onfsm.get_input_alphabet()

    sul = OnfsmSUL(onfsm)
    eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09, reset_after_cex=True)
    eq_oracle = UnseenOutputRandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=4, max_walk_len=10)

    learned_onfsm = run_Lstar_ONFSM(alphabet, sul, eq_oracle, n_sampling=15, print_level=3)

    return learned_onfsm


def abstracted_onfsm_example():
    """
    Learning an abstracted ONFSM. The original ONFSM has 9 states.
    The learned abstracted ONFSM only has 3 states.

    :return: learned abstracted ONFSM
    """
    from aalpy.SULs import OnfsmSUL
    from aalpy.oracles import UnseenOutputRandomWalkEqOracle
    from aalpy.learning_algs import run_abstracted_Lstar_ONFSM
    from aalpy.utils import get_ONFSM

    onfsm = get_ONFSM()

    alphabet = onfsm.get_input_alphabet()

    sul = OnfsmSUL(onfsm)
    eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.5, reset_after_cex=True)

    abstraction_mapping = {0: 0, 'O': 0}

    learned_onfsm = run_abstracted_Lstar_ONFSM(alphabet, sul, eq_oracle=eq_oracle, abstraction_mapping=abstraction_mapping,
                                               n_sampling=50, print_level=3)

    return learned_onfsm


def faulty_coffee_machine_mdp_example(automaton_type='mdp'):
    """
    Learning faulty coffee machine that can be found in Chapter 5 and Chapter 7 of Martin's Tappler PhD thesis.
    :automaton_type either mdp or smm
    :return learned MDP
    """
    from aalpy.SULs import MdpSUL
    from aalpy.oracles import UnseenOutputRandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import get_faulty_coffee_machine_MDP

    mdp = get_faulty_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)

    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=500, reset_prob=0.11,
                                               reset_after_cex=False)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, automaton_type=automaton_type,
                                       eq_oracle=eq_oracle, n_c=20, n_resample=100, min_rounds=10,
                                       max_rounds=50, print_level=3, cex_processing='longest_prefix',
                                       samples_cex_strategy='bfs')

    return learned_mdp


def weird_coffee_machine_mdp_example():
    """
    Learning faulty coffee machine that can be found in Chapter 5 and Chapter 7 of Martin's Tappler PhD thesis.
    :return learned MDP
    """
    from aalpy.SULs import MdpSUL
    from aalpy.oracles import UnseenOutputRandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import get_weird_coffee_machine_MDP

    mdp = get_weird_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)

    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=4000, reset_prob=0.11,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=20, n_resample=1000, min_rounds=10,
                                       max_rounds=500, strategy='normal', cex_processing=None,
                                       samples_cex_strategy='bfs', automaton_type='mdp')

    return learned_mdp


def benchmark_stochastic_example(example, automaton_type='smm', n_c=20, n_resample=1000, min_rounds=10, max_rounds=500,
                                 strategy='normal', cex_processing=None, error_bound=None, samples_cex_strategy=None):
    """
    Learning the stochastic Mealy Machine(SMM) various benchmarking examples
    found in Chapter 7 of Martin's Tappler PhD thesis.
    :param n_c: cutoff for a state to be considered complete
    :param automaton_type: either smm (stochastic mealy machine) or mdp (Markov decision process)
    :param n_resample: resampling size
    :param example: One of ['first_grid', 'second_grid', 'shared_coin', 'slot_machine']
    :param min_rounds: minimum number of learning rounds
    :param max_rounds: maximum number of learning rounds
    :param strategy: normal, classic or chi2
    :param cex_processing: counterexample processing strategy
    :param samples_cex_strategy: strategy to sample cex in the trace tree
    :return: learned SMM
    """
    from aalpy.SULs import StochasticMealySUL
    from aalpy.oracles import UnseenOutputRandomWalkEqOracle, UnseenOutputRandomWordEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import load_automaton_from_file

    mdp = load_automaton_from_file(f'./DotModels/MDPs/{example}.dot', automaton_type='mdp')
    input_alphabet = mdp.get_input_alphabet()

    sul = StochasticMealySUL(mdp)
    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=200, reset_prob=0.25,
                                               reset_after_cex=True)
    eq_oracle = UnseenOutputRandomWordEqOracle(input_alphabet, sul, num_walks=150, min_walk_len=5, max_walk_len=15,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul, n_c=n_c,
                                       n_resample=n_resample, min_rounds=min_rounds, max_rounds=max_rounds,
                                       automaton_type=automaton_type, strategy=strategy, cex_processing=cex_processing,
                                       samples_cex_strategy=samples_cex_strategy, target_unambiguity=0.99,
                                       error_bound=error_bound, property_stop_exp_name=example if error_bound else None)

    return learned_mdp


def custom_smm_example(smm, n_c=20, n_resample=100, min_rounds=10, max_rounds=500):
    """
    Learning custom SMM.
    :param smm: stochastic Mealy machine to learn
    :param n_c: cutoff for a state to be considered complete
    :param n_resample: resampling size
    :param min_rounds: minimum number of learning rounds
    :param max_rounds: maximum number of learning rounds
    :return: learned SMM
    """
    from aalpy.SULs import StochasticMealySUL
    from aalpy.oracles import UnseenOutputRandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar

    input_al = smm.get_input_alphabet()

    sul = StochasticMealySUL(smm)

    eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet=input_al, sul=sul, num_steps=5000, reset_prob=0.2,
                                               reset_after_cex=True)

    learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle, n_c=n_c, n_resample=n_resample,
                                         automaton_type='smm', min_rounds=min_rounds, max_rounds=max_rounds,
                                         print_level=3)

    return learned_model
