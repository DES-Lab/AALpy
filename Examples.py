import string

from aalpy.learning_algs import run_Lstar, run_Lstar_ONFSM, run_stochastic_Lstar
from aalpy.oracles import RandomWalkEqOracle, StatePrefixEqOracle, TransitionFocusOracle, WMethodEqOracle, \
    BreadthFirstExplorationEqOracle, UnseenOutputRandomWalkEqOracle
from aalpy.oracles.CacheBasedEqOracle import CacheBasedEqOracle
from aalpy.oracles.RandomWordEqOracle import RandomWordEqOracle, UnseenOutputRandomWordEqOracle
from aalpy.oracles.UserInputEqOracle import UserInputEqOracle
from aalpy.oracles.WMethodEqOracle import RandomWMethodEqOracle
from aalpy.oracles.kWayStateCoverageEqOracle import KWayStateCoverageEqOracle
from aalpy.SULs import MealySUL, MooreSUL, DfaSUL, TomitaSUL, RegexSUL, FunctionDecorator, PyClassSUL, \
    OnfsmSUL, MdpSUL, StochasticMealySUL
from aalpy.utils.AutomatonGenerators import generate_random_mealy_machine, generate_random_moore_machine, \
    generate_random_dfa, generate_random_ONFSM, generate_random_mdp
from aalpy.utils import MockMqttExample, get_faulty_coffee_machine_MDP, get_benchmark_ONFSM, \
    get_Angluin_dfa, get_weird_coffee_machine_MDP
from aalpy.utils.FileHandler import visualize_automaton, load_automaton_from_file


def random_mealy_example(alphabet_size, number_of_states, output_size=8):
    """
    Generate a random Mealy machine and learn it.
    :param alphabet_size: size of input alphabet
    :param number_of_states: number of states in generated Mealy machine
    :param output_size: size of the output
    :return: learned Mealy machine
    """
    alphabet = [*range(0, alphabet_size)]

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
    assert num_accepting_states <= number_of_states
    alphabet = list(string.ascii_letters[:26])[:alphabet_size]
    random_dfa = generate_random_dfa(number_of_states, alphabet, num_accepting_states)
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
    :param num_states: nubmer of states in generated MDP
    :param input_len: size of input alphabet
    :param n_c: cutoff for a state to be considered complete
    :param n_resample: resampling size
    :param num_outputs: size of output alphabet
    :param min_rounds: minimum number of learning rounds
    :param max_rounds: maximum number of learning rounds
    :return: learned MDP
    """
    mdp, input_alphabet = generate_random_mdp(num_states, input_len, num_outputs)
    sul = MdpSUL(mdp)
    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=5000, reset_prob=0.11,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=n_c, n_resample=n_resample,
                                       min_rounds=min_rounds, max_rounds=max_rounds)

    return learned_mdp


def angluin_seminal_example():
    """
    Example automaton from Anguin's seminal paper.
    :return: learned DFA
    """
    dfa = get_Angluin_dfa()

    alph = dfa.get_input_alphabet()

    sul = DfaSUL(dfa)
    eq_oracle = RandomWalkEqOracle(alph, sul, 500)

    learned_dfa = run_Lstar(alph, sul, eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=True, cex_processing=None, print_level=3)

    return learned_dfa


def tomita_example(tomita_number):
    """
    Pass a tomita function to this example and learn it.
    :param: function of the desired tomita grammar
    :rtype: Dfa
    :return DFA representing tomita grammar
    """
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
    input_al = ['connect', 'disconnect', 'publish','subscribe', 'unsubscribe']

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
    seed(3)
    onfsm = get_benchmark_ONFSM()

    alph = onfsm.get_input_alphabet()

    sul = OnfsmSUL(onfsm)
    eq_oracle = UnseenOutputRandomWalkEqOracle(alph, sul, num_steps=5000, reset_prob=0.09, reset_after_cex=True)
    eq_oracle = UnseenOutputRandomWordEqOracle(alph, sul, num_walks=500, min_walk_len=4, max_walk_len=10)

    learned_onfsm = run_Lstar_ONFSM(alph, sul, eq_oracle, n_sampling=15, print_level=3)

    return learned_onfsm


def faulty_coffee_machine_mdp_example():
    """
    Learning faulty coffee machine that can be found in Chapter 5 and Chapter 7 of Martin's Tappler PhD thesis.
    :return learned MDP
    """
    mdp = get_faulty_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)
    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=1000, reset_prob=0.11,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=20, n_resample=100, min_rounds=10,
                                       max_rounds=50, print_level=3)

    return learned_mdp


def weird_coffee_machine_mdp_example():
    """
    Learning faulty coffee machine that can be found in Chapter 5 and Chapter 7 of Martin's Tappler PhD thesis.
    :return learned MDP
    """
    mdp = get_weird_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)

    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=4000, reset_prob=0.11,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=20, n_resample=100, min_rounds=10,
                                       max_rounds=500, strategy='no_cq')

    return learned_mdp


def benchmark_mdp_example(example='first_grid', n_c=20, n_resample=1000, min_rounds=10, max_rounds=500,
                          strategy='normal'):
    """
    Learning the MDP various benchmarking examples found in Chapter 7 of Martin's Tappler PhD thesis.
    :param example: One of ['first_grid', 'second_grid', 'shared_coin', 'slot_machine']
    :param n_c: cutoff for a state to be considered complete
    :param n_resample: resampling size
    :param min_rounds: minimum number of learning rounds
    :param max_rounds: maximum number of learning rounds
    :param strategy: normal or no_cq
    :return: learned MDP
    """
    mdp = load_automaton_from_file(f'./DotModels/MDPs/{example}.dot', automaton_type='mdp')

    input_alphabet = mdp.get_input_alphabet()
    output_alphabet = list({state.output for state in mdp.states})
    sul = MdpSUL(mdp)
    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=5000, reset_prob=0.09,
                                               reset_after_cex=True)
    eq_oracle = UnseenOutputRandomWordEqOracle(input_alphabet, sul, num_walks=500, min_walk_len=10, max_walk_len=100,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet=input_alphabet, sul=sul, eq_oracle=eq_oracle,
                                       n_c=n_c, n_resample=n_resample, min_rounds=min_rounds, max_rounds=max_rounds,
                                       strategy=strategy)

    return learned_mdp


def benchmark_mdp_2_smm_example(example='first_grid', n_c=20, n_resample=1000, min_rounds=10, max_rounds=500,
                                strategy='normal'):
    """
    Learning the stochastic Mealy Machine(SMM) various benchmarking examples
    found in Chapter 7 of Martin's Tappler PhD thesis.
    :param n_c: cutoff for a state to be considered complete
    :param n_resample: resampling size
    :param example: One of ['first_grid', 'second_grid', 'shared_coin', 'slot_machine']
    :param min_rounds: minimum number of learning rounds
    :param max_rounds: maximum number of learning rounds
    :param strategy: normal or no_cq
    :return: learned SMM
    """
    mdp = load_automaton_from_file(f'./DotModels/MDPs/{example}.dot', automaton_type='mdp')
    input_alphabet = mdp.get_input_alphabet()

    sul = StochasticMealySUL(mdp)
    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, sul=sul, num_steps=5000, reset_prob=0.09,
                                               reset_after_cex=True)
    eq_oracle = UnseenOutputRandomWordEqOracle(input_alphabet, sul, num_walks=150, min_walk_len=10, max_walk_len=300,
                                               reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul, n_c=n_c,
                                       n_resample=n_resample, min_rounds=min_rounds, max_rounds=max_rounds,
                                       automaton_type='smm', strategy=strategy)

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
    input_al = smm.get_input_alphabet()

    sul = StochasticMealySUL(smm)

    eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet=input_al, sul=sul, num_steps=5000, reset_prob=0.2,
                                               reset_after_cex=True)

    learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle, n_c=n_c, n_resample=n_resample,
                                         automaton_type='smm', min_rounds=min_rounds, max_rounds=max_rounds,
                                         print_level=3)

    return learned_model
