def random_deterministic_model_example():
    from aalpy.utils import generate_random_deterministic_automata
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWMethodEqOracle
    from aalpy.learning_algs import run_KV

    model_type = 'mealy'  # or 'moore', 'dfa'

    # for random dfa's you can also define num_accepting_states
    random_model = generate_random_deterministic_automata(automaton_type=model_type, num_states=100,
                                                          input_alphabet_size=3, output_alphabet_size=4)

    sul = AutomatonSUL(random_model)
    input_alphabet = random_model.get_input_alphabet()

    # select any of the oracles
    eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=10, walk_len=20)

    learned_model = run_KV(input_alphabet, sul, eq_oracle, model_type)

    assert learned_model == random_model
    return learned_model


def angluin_seminal_example():
    """
    Example automaton from Angluin's seminal paper.
    :return: learned DFA
    """
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import get_Angluin_dfa

    dfa = get_Angluin_dfa()

    alphabet = dfa.get_input_alphabet()

    sul = AutomatonSUL(dfa)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500)

    learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=True, cex_processing=None, print_level=3)

    assert learned_dfa == dfa
    return learned_dfa

def angluin_seminal_example_lsharp():
    """
    Example automaton from Angluin's seminal paper.
    :return: learned DFA
    """
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_Lstar, run_Lsharp
    from aalpy.utils import get_Angluin_dfa

    dfa = get_Angluin_dfa()

    alphabet = dfa.get_input_alphabet()

    sul = AutomatonSUL(dfa)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, 500)

    learned_dfa = run_Lsharp(alphabet, sul, eq_oracle, automaton_type='dfa',
                            extension_rule="SepSeq", separation_rule="ADS", max_learning_rounds=50, print_level=3)

    assert learned_dfa == dfa
    return learned_dfa


def tomita_example(tomita_number=3):
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
    state_origin_eq_oracle = StatePrefixEqOracle(alphabet, tomita_sul, walks_per_state=50, walk_len=10)

    # or replace run_Lstar with run_KV
    learned_dfa = run_Lstar(alphabet, tomita_sul, state_origin_eq_oracle, automaton_type='dfa', )

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

    eq_oracle = StatePrefixEqOracle(alphabet, regex_sul, walks_per_state=2000,
                                    walk_len=15)

    # or replace run_Lstar with run_KV
    learned_regex = run_Lstar(alphabet, regex_sul, eq_oracle, automaton_type='dfa')

    return learned_regex


def learn_date_validator():
    from aalpy.base import SUL
    from aalpy.utils import DateValidator
    from aalpy.oracles import StatePrefixEqOracle
    from aalpy.learning_algs import run_Lstar

    class DateSUL(SUL):
        """
        An example implementation of a system under learning that
        can be used to learn the behavior of the date validator.
        """

        def __init__(self):
            super().__init__()
            # DateValidator is a black-box class used for date string verification
            # The format of the dates is %d/%m/%Y'
            # Its method is_date_accepted returns True if date is accepted, False otherwise
            self.dv = DateValidator()
            self.string = ""

        def pre(self):
            # reset the string used for testing
            self.string = ""
            pass

        def post(self):
            pass

        def step(self, letter):
            # add the input to the current string
            if letter is not None:
                self.string += str(letter)

            # test if the current sting is accepted
            return self.dv.is_date_accepted(self.string)

    # instantiate the SUL
    sul = DateSUL()

    # define the input alphabet
    alphabet = list(range(0, 9)) + ['/']

    # define a equivalence oracle

    eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=500, walk_len=15)

    # run the learning algorithm

    learned_model = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa')
    # visualize the automaton
    learned_model.visualize()


def bluetooth_Lsharp():
    from aalpy.utils import load_automaton_from_file
    from aalpy.SULs import MealySUL
    from aalpy.oracles import WpMethodEqOracle
    from aalpy.learning_algs import run_Lsharp

    mealy_machine = load_automaton_from_file(f'./DotModels/Bluetooth/CYW43455.dot', automaton_type='mealy')
    input_alphabet = mealy_machine.get_input_alphabet()

    sul_mealy = MealySUL(mealy_machine)
    eq_oracle = WpMethodEqOracle(input_alphabet, sul_mealy, len(mealy_machine.states))

    # Extension rule options: {"Nothing", "SepSeq", "ADS"}
    # Separation rule options: {"SepSeq", "ADS"}
    learned_mealy = run_Lsharp(input_alphabet, sul_mealy, eq_oracle, automaton_type='mealy', extension_rule=None,
                               separation_rule="SepSeq", max_learning_rounds=50, print_level=3)


def bluetooth_adaptive_Lsharp():
    from aalpy.utils import load_automaton_from_file
    from aalpy.SULs import MealySUL
    from aalpy.oracles import WpMethodEqOracle
    from aalpy.learning_algs import run_adaptive_Lsharp

    reference1 = load_automaton_from_file(f'./DotModels/Bluetooth/CC2650.dot', automaton_type='mealy')
    reference2 = load_automaton_from_file(f'./DotModels/Bluetooth/nRF52832.dot', automaton_type='mealy')
    reference3 = load_automaton_from_file(f'./DotModels/Bluetooth/CC2640R2-no-feature-req.dot', automaton_type='mealy')
    target = load_automaton_from_file(f'./DotModels/Bluetooth/CYW43455.dot', automaton_type='mealy')

    input_alphabet = target.get_input_alphabet()

    sul_mealy = MealySUL(target)
    eq_oracle = WpMethodEqOracle(input_alphabet, sul_mealy, len(target.states))

    # Rebuilding options: {True, False}
    # State Matching options: {None, "Total", "Approximate"}
    learned_mealy = run_adaptive_Lsharp(input_alphabet, sul_mealy, [reference1, reference2, reference3], eq_oracle,
                                       automaton_type='mealy', extension_rule='SepSeq', separation_rule="ADS",
                                       rebuilding=True, state_matching="Approximate", print_level=2)



def random_deterministic_example_with_provided_sequences():
    from random import choice, randint
    from aalpy.SULs import AutomatonSUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import generate_random_deterministic_automata

    random_mealy = generate_random_deterministic_automata('mealy', num_states=10,
                                                          input_alphabet_size=4, output_alphabet_size=3)

    input_alphabet = random_mealy.get_input_alphabet()

    sul_mealy = AutomatonSUL(random_mealy)

    # samples obtained form somewhere else
    samples = []
    for _ in range(1000):
        inputs = tuple(choice(input_alphabet) for _ in range(randint(4, 12)))
        outputs = sul_mealy.query(inputs)
        input_output_pair = (inputs, outputs)
        samples.append(input_output_pair)

    from aalpy.oracles import RandomWalkEqOracle
    random_walk_eq_oracle = RandomWalkEqOracle(input_alphabet, sul_mealy, 5000)

    learned_mealy = run_Lstar(input_alphabet, sul_mealy, random_walk_eq_oracle, automaton_type='mealy',
                              samples=samples)


def big_input_alphabet_example():
    """
        Small example where input alphabet can be huge and outputs are just true and false (DFA).
    """

    from aalpy.base import SUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import RandomWMethodEqOracle

    input_alphabet_size = 1000
    automaton_depth = 4

    class alternatingSUL(SUL):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def pre(self):
            self.counter = 0

        def post(self):
            pass

        def step(self, letter):
            if letter is None:
                return False
            out = letter % 2
            self.counter = min(self.counter + 1, automaton_depth)
            if self.counter % 2 == 1:
                return not out
            return out

    input_al = list(range(input_alphabet_size))

    sul = alternatingSUL()
    eq_oracle = RandomWMethodEqOracle(input_al, sul)

    model = run_Lstar(input_al, sul, eq_oracle, 'dfa', cache_and_non_det_check=False)

    return model


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
    from aalpy.SULs import AutomatonSUL
    from aalpy.utils import generate_random_ONFSM
    from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
    from aalpy.learning_algs import run_non_det_Lstar

    onfsm = generate_random_ONFSM(num_states=num_states, num_inputs=input_size, num_outputs=output_size)
    alphabet = onfsm.get_input_alphabet()

    sul = AutomatonSUL(onfsm)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=500, reset_prob=0.15, reset_after_cex=True)
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=8, max_walk_len=20)

    learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle=eq_oracle, n_sampling=n_sampling, print_level=2)
    return learned_model


def random_mdp_example():
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import generate_random_mdp

    mdp = generate_random_mdp(num_states=10, input_size=3, output_size=3)
    input_alphabet = mdp.get_input_alphabet()
    sul = AutomatonSUL(mdp)

    eq_oracle = RandomWalkEqOracle(input_alphabet, sul=sul, num_steps=5000, reset_prob=0.11,
                                   reset_after_cex=False)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle,
                                       min_rounds=5, max_rounds=50)

    return learned_mdp


def learn_python_class():
    """
    Learn a Mealy machine where inputs are methods and arguments of the class that serves as SUL.
    :return: Mealy machine
    """

    # class
    from aalpy.SULs import PyClassSUL, FunctionDecorator
    from aalpy.oracles import StatePrefixEqOracle
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import MockMqttExample

    mqtt = MockMqttExample

    input_al = [FunctionDecorator(mqtt.connect), FunctionDecorator(mqtt.disconnect),
                FunctionDecorator(mqtt.subscribe, 'topic'), FunctionDecorator(mqtt.unsubscribe, 'topic'),
                FunctionDecorator(mqtt.publish, 'topic')]

    sul = PyClassSUL(mqtt)

    eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=20, walk_len=20)

    mealy = run_Lstar(input_al, sul, eq_oracle=eq_oracle, automaton_type='mealy', cache_and_non_det_check=True)

    mealy.visualize()


def mqtt_example():
    from aalpy.base import SUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import MockMqttExample

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

    mealy.visualize()


def onfsm_mealy_paper_example():
    """
    Learning a ONFSM presented in 'Learning Finite State Models of Observable Nondeterministic Systems in a Testing
    Context'.
    :return: learned ONFSM
    """

    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_non_det_Lstar
    from aalpy.utils import get_benchmark_ONFSM

    onfsm = get_benchmark_ONFSM()
    alphabet = onfsm.get_input_alphabet()

    sul = AutomatonSUL(onfsm)
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=5, max_walk_len=12)

    learned_onfsm = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10, print_level=2)

    return learned_onfsm


def multi_client_mqtt_example():
    """
    Example from paper P'Learning Abstracted Non-deterministic Finite State Machines'.
    https://link.springer.com/chapter/10.1007/978-3-030-64881-7_4

    Returns:

        learned automaton
    """
    import random

    from aalpy.base import SUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_abstracted_ONFSM_Lstar
    from aalpy.SULs import AutomatonSUL
    from aalpy.utils import load_automaton_from_file

    class Multi_Client_MQTT_Mapper(SUL):
        def __init__(self):
            super().__init__()

            five_clients_mqtt_mealy = load_automaton_from_file('DotModels/five_clients_mqtt_abstracted_onfsm.dot',
                                                               automaton_type='mealy')
            self.five_client_mqtt = AutomatonSUL(five_clients_mqtt_mealy)
            self.connected_clients = set()
            self.subscribed_clients = set()

            self.clients = ('c0', 'c1', 'c2', 'c3', 'c4')

        def get_input_alphabet(self):
            return ['connect', 'disconnect', 'subscribe', 'unsubscribe', 'publish']

        def pre(self):
            self.five_client_mqtt.pre()

        def post(self):
            self.five_client_mqtt.post()
            self.connected_clients = set()
            self.subscribed_clients = set()

        def step(self, letter):
            client = random.choice(self.clients)
            inp = client + '_' + letter
            concrete_output = self.five_client_mqtt.step(inp)
            all_out = ''

            if letter == 'connect':
                if client not in self.connected_clients:
                    self.connected_clients.add(client)
                elif client in self.connected_clients:
                    self.connected_clients.remove(client)
                    if client in self.subscribed_clients:
                        self.subscribed_clients.remove(client)
                    if len(self.subscribed_clients) == 0:
                        all_out = '_UNSUB_ALL'

            elif letter == 'subscribe' and client in self.connected_clients:
                self.subscribed_clients.add(client)
            elif letter == 'disconnect' and client in self.connected_clients:
                self.connected_clients.remove(client)
                if client in self.subscribed_clients:
                    self.subscribed_clients.remove(client)
                if len(self.subscribed_clients) == 0:
                    all_out = '_UNSUB_ALL'
            elif letter == 'unsubscribe' and client in self.connected_clients:
                if client in self.subscribed_clients:
                    self.subscribed_clients.remove(client)
                if len(self.subscribed_clients) == 0:
                    all_out = '_ALL'

            concrete_outputs = concrete_output.split('__')
            abstract_outputs = set([e[3:] for e in concrete_outputs])
            if 'Empty' in abstract_outputs:
                abstract_outputs.remove('Empty')
            if abstract_outputs == {'CONCLOSED'}:
                if len(self.connected_clients) == 0:
                    all_out = '_ALL'
                return 'CONCLOSED' + all_out
            else:
                if 'CONCLOSED' in abstract_outputs:
                    abstract_outputs.remove('CONCLOSED')
                abstract_outputs = sorted(list(abstract_outputs))
                output = '_'.join(abstract_outputs)
                return '_'.join(set(output.split('_'))) + all_out

    sul = Multi_Client_MQTT_Mapper()
    alphabet = sul.get_input_alphabet()

    eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09, reset_after_cex=True)

    abstraction_mapping = {
        'CONCLOSED': 'CONCLOSED',
        'CONCLOSED_UNSUB_ALL': 'CONCLOSED',
        'CONCLOSED_ALL': 'CONCLOSED',
        'UNSUBACK': 'UNSUBACK',
        'UNSUBACK_ALL': 'UNSUBACK'
    }

    learned_onfsm = run_abstracted_ONFSM_Lstar(alphabet, sul, eq_oracle, abstraction_mapping=abstraction_mapping,
                                               n_sampling=200, print_level=3)

    return learned_onfsm


def abstracted_onfsm_example():
    """
    Learning an abstracted ONFSM. The original ONFSM has 9 states.
    The learned abstracted ONFSM only has 3 states.

    :return: learned abstracted ONFSM
    """
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_abstracted_ONFSM_Lstar
    from aalpy.utils import get_ONFSM

    onfsm = get_ONFSM()

    alphabet = onfsm.get_input_alphabet()

    sul = AutomatonSUL(onfsm)
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=4, max_walk_len=8, reset_after_cex=True)

    abstraction_mapping = {0: 0, 'O': 0}

    learned_onfsm = run_abstracted_ONFSM_Lstar(alphabet, sul, eq_oracle=eq_oracle,
                                               abstraction_mapping=abstraction_mapping,
                                               n_sampling=50, print_level=3)

    return learned_onfsm


def faulty_coffee_machine_mdp_example(automaton_type='mdp'):
    """
    Learning faulty coffee machine that can be found in Chapter 5 and Chapter 7 of Martin's Tappler PhD thesis.
    :automaton_type either mdp or smm
    :return learned MDP
    """
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import get_faulty_coffee_machine_MDP

    mdp = get_faulty_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = AutomatonSUL(mdp)

    eq_oracle = RandomWalkEqOracle(input_alphabet, sul=sul, num_steps=500, reset_prob=0.11,
                                   reset_after_cex=False)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, automaton_type=automaton_type,
                                       eq_oracle=eq_oracle, n_c=20, n_resample=100, min_rounds=3,
                                       max_rounds=50, print_level=3, cex_processing='longest_prefix',
                                       samples_cex_strategy='bfs')

    return learned_mdp


def weird_coffee_machine_mdp_example():
    """
    Learning faulty coffee machine that can be found in Chapter 5 and Chapter 7 of Martin's Tappler PhD thesis.
    :return learned MDP
    """
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import get_weird_coffee_machine_MDP

    mdp = get_weird_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = AutomatonSUL(mdp)

    eq_oracle = RandomWordEqOracle(input_alphabet, sul=sul, num_walks=2000, min_walk_len=4, max_walk_len=10,
                                   reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=20, n_resample=1000, min_rounds=10,
                                       max_rounds=500, strategy='normal', cex_processing=None,
                                       samples_cex_strategy=None, automaton_type='smm')

    return learned_mdp


def benchmark_stochastic_example(example, automaton_type='smm', n_c=20, n_resample=1000, min_rounds=10, max_rounds=500,
                                 strategy='normal', cex_processing='longest_prefix', stopping_based_on_prop=None,
                                 samples_cex_strategy=None):
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
    :stopping_based_on_prop: a tuple (path to properties, correct values, error bound)
    :param samples_cex_strategy: strategy to sample cex in the trace tree
    :return: learned SMM

    """
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import load_automaton_from_file

    # Specify the path to the dot file containing a MDP
    mdp = load_automaton_from_file(f'./DotModels/MDPs/{example}.dot', automaton_type='mdp')
    input_alphabet = mdp.get_input_alphabet()

    sul = AutomatonSUL(mdp)
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=5, max_walk_len=15,
                                   reset_after_cex=True)
    eq_oracle = RandomWalkEqOracle(input_alphabet, sul=sul, num_steps=2000, reset_prob=0.25,
                                   reset_after_cex=True)

    learned_mdp = run_stochastic_Lstar(input_alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul, n_c=n_c,
                                       n_resample=n_resample, min_rounds=min_rounds, max_rounds=max_rounds,
                                       automaton_type=automaton_type, strategy=strategy, cex_processing=cex_processing,
                                       samples_cex_strategy=samples_cex_strategy, target_unambiguity=0.99,
                                       property_based_stopping=stopping_based_on_prop)

    return learned_mdp


def custom_stochastic_example(stochastic_machine, learning_type='smm', min_rounds=10, max_rounds=500):
    """
    Learning custom SMM.
    :param stochastic_machine: stochastic Mealy machine or MDP to learn
    :param learning_type: 'smm' or 'mdp'
    :param min_rounds: minimum number of learning rounds
    :param max_rounds: maximum number of learning rounds
    :return: learned model
    """
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar

    input_al = stochastic_machine.get_input_alphabet()

    sul = AutomatonSUL(stochastic_machine)

    eq_oracle = RandomWordEqOracle(alphabet=input_al, sul=sul, num_walks=1000, min_walk_len=10, max_walk_len=30,
                                   reset_after_cex=True)

    learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle,
                                         automaton_type=learning_type,
                                         min_rounds=min_rounds,
                                         max_rounds=max_rounds,
                                         print_level=2)

    return learned_model


def learn_stochastic_system_and_do_model_checking(example, automaton_type='smm', n_c=20, n_resample=1000, min_rounds=10,
                                                  max_rounds=500, strategy='normal', cex_processing='longest_prefix',
                                                  stopping_based_on_prop=None, samples_cex_strategy=None):
    import aalpy.paths
    from aalpy.automata import StochasticMealyMachine
    from aalpy.utils import model_check_experiment, get_properties_file, get_correct_prop_values
    from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

    aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
    aalpy.paths.path_to_properties = "Benchmarking/prism_eval_props/"

    learned_model = benchmark_stochastic_example(example, automaton_type, n_c, n_resample, min_rounds, max_rounds,
                                                 strategy,
                                                 cex_processing, stopping_based_on_prop, samples_cex_strategy)

    if isinstance(learned_model, StochasticMealyMachine):
        mdp = smm_to_mdp_conversion(learned_model)
    else:
        mdp = learned_model

    values, diff = model_check_experiment(get_properties_file(example), get_correct_prop_values(example), mdp)

    print('Value for each property:', [round(d * 100, 2) for d in values.values()])
    print('Error for each property:', [round(d * 100, 2) for d in diff.values()])


def alergia_mdp_example():
    from aalpy.SULs import AutomatonSUL
    from random import randint, choice
    from aalpy.learning_algs import run_Alergia
    from aalpy.utils import generate_random_mdp

    mdp = generate_random_mdp(5, 2, 3)
    initial_output = mdp.initial_state.output
    sul = AutomatonSUL(mdp)
    inputs = mdp.get_input_alphabet()

    data = []
    for _ in range(100000):
        str_len = randint(5, 12)
        # add the initial output
        sul.pre()
        seq = [initial_output]
        for _ in range(str_len):
            i = choice(inputs)
            o = sul.step(i)
            seq.append((i, o))
        sul.post()
        data.append(seq)

    # run alergia with the data and automaton_type set to 'mdp' to True to learn a MDP
    model = run_Alergia(data, automaton_type='mdp', eps=0.05, print_info=True)

    model.visualize()


def alergia_smm_example():
    from aalpy.SULs import AutomatonSUL
    from random import randint, choice
    from aalpy.learning_algs import run_Alergia
    from aalpy.utils import generate_random_smm

    smm = generate_random_smm(5, 2, 2)
    sul = AutomatonSUL(smm)
    inputs = smm.get_input_alphabet()

    data = []
    for _ in range(100000):
        str_len = randint(5, 15)
        sul.pre()
        seq = []
        for _ in range(str_len):
            i = choice(inputs)
            o = sul.step(i)
            seq.append((i, o))
        sul.post()
        data.append(seq)

    # run alergia with the data and automaton_type set to 'mdp' to True to learn a MDP
    model = run_Alergia(data, automaton_type='smm', eps=0.05, print_info=True)

    model.visualize()
    return model


def alergia_mc_example_with_loaded_data():
    from os import remove
    from aalpy.SULs import AutomatonSUL
    from random import randint
    from aalpy.learning_algs import run_Alergia
    from aalpy.utils import generate_random_markov_chain
    from aalpy.utils import CharacterTokenizer

    mc = generate_random_markov_chain(10)
    initial_output = mc.initial_state.output
    mc.visualize('Original')

    sul = AutomatonSUL(mc)

    # note that this example shows writing to file just to show how tokenizer is used...
    # this step can ofc be skipped and lists passed to alergia
    data = []
    for _ in range(20000):
        sul.pre()
        str_len = randint(4, 12)
        seq = [f'{initial_output}']
        for _ in range(str_len):
            o = sul.step()
            seq.append(f'{o}')
        sul.post()
        data.append(''.join(seq))

    with open('mcData.txt', 'w') as file:
        for seq in data:
            file.write(f'{seq}\n')

    file.close()

    # create tokenizer
    tokenizer = CharacterTokenizer()
    # parse data
    data = tokenizer.tokenize_data('mcData.txt')
    # run alergia with the data and automaton_type set to 'mc' to learn a Markov Chain
    model = run_Alergia(data, automaton_type='mc', eps=0.05, print_info=True)
    # print(model)

    model.visualize()
    remove('mcData.txt')
    return model


def jAlergiaExample():
    from aalpy.learning_algs import run_JAlergia

    # if you need more heap space check
    model = run_JAlergia(path_to_data_file='jAlergia/exampleMdpData.txt', automaton_type='mdp', eps=0.05,
                         path_to_jAlergia_jar='jAlergia/alergia.jar')

    # # alternatively pass the data in following format
    # mc_data = [[1,2,3,4,5], [1,2,3,4,2,1], [1,3,5,2,3]]
    # mdp_data = [[1,2,3,1,2], [1,3,6,4,2]]
    # model = run_JAlergia(path_to_data_file=mc_data, automaton_type='mdp', eps=0.05,
    #                      path_to_jAlergia_jar='jAlergia/alergia.jar', optimize_for='memory')

    model.visualize()
    return model


def active_alergia_example(example='first_grid'):
    from random import choice, randint
    from aalpy.SULs import AutomatonSUL
    from aalpy.utils import load_automaton_from_file
    from aalpy.learning_algs import run_active_Alergia
    from aalpy.learning_algs.stochastic_passive.ActiveAleriga import RandomWordSampler

    mdp = load_automaton_from_file(f'./DotModels/MDPs/{example}.dot', automaton_type='mdp')
    input_alphabet = mdp.get_input_alphabet()

    sul = AutomatonSUL(mdp)

    data = []
    for _ in range(50000):
        input_query = tuple(choice(input_alphabet) for _ in range(randint(6, 14)))
        outputs = sul.query(input_query)
        # format data in [O, (I, O), (I, O)...]
        formatted_io = [outputs.pop(0)]
        for i, o in zip(input_query, outputs):
            formatted_io.append((i, o))
        data.append(formatted_io)

    sampler = RandomWordSampler(num_walks=1000, min_walk_len=8, max_walk_len=20)
    model = run_active_Alergia(data, sul, sampler, n_iter=10)

    print(model)


def rpni_example():
    data = [(('a', 'a', 'a'), True),
            (('a', 'a', 'b', 'a'), True),
            (('b', 'b', 'a'), True),
            (('b', 'b', 'a', 'b', 'a'), True),
            (('a',), False),
            (('b', 'b'), False),
            (('a', 'a', 'b'), False),
            (('a', 'b', 'a'), False)]

    from aalpy.learning_algs import run_RPNI
    model = run_RPNI(data, automaton_type='dfa')
    model.visualize()


def rpni_check_model_example():
    from aalpy.SULs import AutomatonSUL
    from aalpy.learning_algs import run_RPNI
    from aalpy.oracles import StatePrefixEqOracle
    from aalpy.utils import generate_random_moore_machine
    from aalpy.utils import generate_input_output_data_from_automata, convert_i_o_traces_for_RPNI

    # model = generate_random_dfa(num_states=5, alphabet=[1, 2, 3], num_accepting_states=2)
    model = generate_random_moore_machine(num_states=5, input_alphabet=[1, 2, 3], output_alphabet=['a', 'b'])

    input_al = model.get_input_alphabet()

    data = generate_input_output_data_from_automata(model, num_sequances=2000,
                                                    min_seq_len=1, max_seq_len=12)

    data = convert_i_o_traces_for_RPNI(data)

    rpni_model = run_RPNI(data, automaton_type='moore', print_info=True)

    rpni_model.make_input_complete('sink_state')
    sul = AutomatonSUL(model)
    eq_oracle_2 = StatePrefixEqOracle(input_al, sul, walks_per_state=100)
    cex = eq_oracle_2.find_cex(rpni_model)

    # or simply do
    # if rpni_model != model
    if cex is None:
        print("Could not find a counterexample between the RPNI-model and the original model.")
    else:
        print('Counterexample found. Either RPNI data was incomplete (or there is a bug in RPNI algorithm :o )')


def rpni_mealy_example():
    import random
    from aalpy.learning_algs import run_RPNI
    from aalpy.utils import generate_random_deterministic_automata
    from aalpy.utils.HelperFunctions import all_prefixes
    # make reproducible
    random.seed(1)

    model = generate_random_deterministic_automata(automaton_type='mealy', num_states=5,
                                                   input_alphabet_size=3, output_alphabet_size=4)
    # model = load_automaton_from_file('DotModels/Bluetooth/bluetooth_model.dot', automaton_type='mealy')

    input_al = model.get_input_alphabet()
    num_sequences = 1000
    data = []
    for _ in range(num_sequences):
        seq_len = random.randint(1, 10)
        random_seq = random.choices(input_al, k=seq_len)
        # make sure that all prefixes all included in the dataset
        for prefix in all_prefixes(random_seq):
            output = model.compute_output_seq(model.initial_state, prefix)[-1]
            data.append((prefix, output))

    rpni_model = run_RPNI(data, automaton_type='mealy', print_info=True)

    return rpni_model


def random_active_rpni_example():
    import random
    from aalpy.learning_algs import run_active_RPNI
    from aalpy.learning_algs.deterministic_passive.active_RPNI import RandomWordSampler
    from aalpy.utils import generate_random_deterministic_automata
    from aalpy.utils.HelperFunctions import all_prefixes
    from aalpy.SULs import AutomatonSUL

    model = generate_random_deterministic_automata('mealy', num_states=50,
                                                   input_alphabet_size=3, output_alphabet_size=5)

    input_al = model.get_input_alphabet()
    num_sequences = 100
    data = []
    for _ in range(num_sequences):
        seq_len = random.randint(1, 20)
        random_seq = random.choices(input_al, k=seq_len)
        # make sure that all prefixes all included in the dataset
        for prefix in all_prefixes(random_seq):
            output = model.compute_output_seq(model.initial_state, prefix)[-1]
            data.append((prefix, output))

    sampler = RandomWordSampler(500, 5, 25)
    sul = AutomatonSUL(model)
    active_rpni_model = run_active_RPNI(data, sul, sampler=sampler, n_iter=5,
                                        automaton_type='mealy', print_info=True)

    return active_rpni_model


def compare_stochastic_and_non_deterministic_learning(example='first_grid'):
    import aalpy.paths
    aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
    aalpy.paths.path_to_properties = "Benchmarking/prism_eval_props/"

    from aalpy.SULs import AutomatonSUL
    from aalpy.automata import StochasticMealyMachine
    from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion
    from aalpy.learning_algs import run_stochastic_Lstar, run_non_det_Lstar
    from aalpy.utils import load_automaton_from_file, model_check_experiment, get_properties_file, \
        get_correct_prop_values
    from aalpy.oracles import RandomWordEqOracle

    mdp = load_automaton_from_file(f'./DotModels/MDPs/first_grid.dot', automaton_type='mdp')
    input_alphabet = mdp.get_input_alphabet()

    # Stochastic Learning
    print("Stochastic Learning")
    sul = AutomatonSUL(mdp)
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=5, max_walk_len=15,
                                   reset_after_cex=True)
    stochastic_learned_model = run_stochastic_Lstar(input_alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul,
                                                    automaton_type='smm', target_unambiguity=0.99)

    # Non Deterministic Learning
    print("Non-deterministic Learning")
    sul = AutomatonSUL(mdp)
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=5, max_walk_len=15,
                                   reset_after_cex=True)
    non_det_model = run_non_det_Lstar(alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul, n_sampling=5,
                                      stochastic=True, print_level=2)

    for model_type, model in [('Stochastic Learning', stochastic_learned_model),
                              ('Non-deterministic learning', non_det_model)]:
        if isinstance(model, StochasticMealyMachine):
            mdp = smm_to_mdp_conversion(model)
        else:
            mdp = model

    values, diff = model_check_experiment(get_properties_file(example), get_correct_prop_values(example), mdp)

    print(model_type)
    print('Error for each property:', [round(d * 100, 2) for d in diff.values()])


def learning_context_free_grammar_example():
    from aalpy.automata import SevpaAlphabet
    from aalpy.learning_algs import run_KV
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.utils.BenchmarkSULs import get_balanced_string_sul

    call_return_map = {'(': ')', '[': ']'}

    sevpa_alphabet = SevpaAlphabet([], list(call_return_map.keys()), list(call_return_map.values()))

    # bounded deterministic approximation
    balanced_string_sul = get_balanced_string_sul(call_return_map, allow_empty_string=False)
    eq_oracle = RandomWordEqOracle(sevpa_alphabet.get_merged_alphabet(), balanced_string_sul, num_walks=1000,
                                   min_walk_len=5, max_walk_len=30)

    learned_deterministic_approximation = run_KV(sevpa_alphabet.get_merged_alphabet(),
                                                 balanced_string_sul, eq_oracle, automaton_type='dfa',
                                                 max_learning_rounds=20)

    balanced_string_sul = get_balanced_string_sul(call_return_map, allow_empty_string=False)
    eq_oracle = RandomWordEqOracle(sevpa_alphabet.get_merged_alphabet(), balanced_string_sul, num_walks=1000,
                                   min_walk_len=5, max_walk_len=30)
    learned_model = run_KV(sevpa_alphabet, balanced_string_sul, eq_oracle, automaton_type='vpa')
    learned_model.visualize()


def arithmetic_expression_sevpa_learning():
    from aalpy.base import SUL
    from aalpy.automata import SevpaAlphabet
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_KV
    import warnings
    import ast
    warnings.filterwarnings("ignore")

    class ArithmeticSUL(SUL):
        def __init__(self):
            super().__init__()
            self.string_under_test = ''

        def pre(self):
            self.string_under_test = ''

        def post(self):
            pass

        def step(self, letter):
            if letter:
                self.string_under_test += ' ' + letter if len(self.string_under_test) > 0 else letter

            try:
                # Parse the expression using ast
                parsed_expr = ast.parse(self.string_under_test, mode='eval')
                # Check if the parsed expression is a valid arithmetic expression
                is_valid = all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Name, ast.Load))
                               or isinstance(node, ast.operator) or isinstance(node, ast.expr_context)
                               or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.operator))
                               for node in ast.walk(parsed_expr))
                return is_valid

            except SyntaxError:
                return False

    sul = ArithmeticSUL()

    alphabet = SevpaAlphabet(internal_alphabet=['1', '+'], call_alphabet=['('], return_alphabet=[')'])

    eq_oracle = RandomWordEqOracle(alphabet.get_merged_alphabet(), sul, min_walk_len=2,
                                   max_walk_len=10, num_walks=2000)

    learned_model = run_KV(alphabet, sul, eq_oracle, automaton_type='vpa')

    learned_model.visualize()


def benchmark_sevpa_learning():
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_KV
    from aalpy.utils.BenchmarkSevpaModels import sevpa_for_L1, sevpa_for_L2, sevpa_for_L11, sevpa_for_L12, sevpa_for_L14

    models = [sevpa_for_L1(), sevpa_for_L2(), sevpa_for_L11(), sevpa_for_L12(), sevpa_for_L14()]

    for inx, model in enumerate(models):

        alphabet = model.get_input_alphabet()

        sul = AutomatonSUL(model)

        if inx == 4:
            alphabet.exclusive_call_return_pairs = {'(': ')', '[': ']'}

        eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000,
                                       min_walk_len=10, max_walk_len=30)

        learned_model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                               print_level=2, cex_processing='rs')

        print(learned_model.get_random_accepting_word())


def random_sevpa_learning():
    from aalpy.SULs import AutomatonSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_KV
    from aalpy.utils import generate_random_sevpa

    random_svepa = generate_random_sevpa(num_states=50, internal_alphabet_size=3,
                                         call_alphabet_size=3,
                                         return_alphabet_size=3,
                                         acceptance_prob=0.4,
                                         return_transition_prob=0.5)

    # from aalpy.utils.BenchmarkVpaModels import vpa_for_L11
    # balanced_parentheses = vpa_for_L11()

    alphabet = random_svepa.input_alphabet

    sul = AutomatonSUL(random_svepa)

    eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000,
                                   min_walk_len=10, max_walk_len=30)

    model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                   print_level=2, cex_processing='rs')


def passive_vpa_learning_on_lists():
    from aalpy.learning_algs import run_PAPNI
    from aalpy.automata import VpaAlphabet

    vpa_alphabet = VpaAlphabet(internal_alphabet=['1'], call_alphabet=['(', '['], return_alphabet=[')', ']'])

    list_data = [
        (tuple(), False),
        (('[', '[', ']', ']'), True),
        (('(', '(', ')', ')'), True),

        (('[', '[', ']', ')'), False),
        (('[', '[', ')', ']'), False),
        (('[', ')'), False),
        (('(', ']'), False),
        (('[', '[', ']', '1', ']'), True),
        (('[', '1', '[', ']', '1', ']'), True),
        (('(', '1', '[', ']', '1', ')'), True),
        (('(', ')'), True),
        (('[', ']'), True),
        (('[', '1', '(', ']', '1', ')'), False),
    ]

    papni = run_PAPNI(list_data, vpa_alphabet, algorithm='gsm', print_info=True)
    papni.visualize()


def passive_vpa_learning_arithmetics():
    from aalpy.learning_algs import run_PAPNI
    from aalpy.utils.BenchmarkVpaModels import gen_arithmetic_data
    arithmetic_data, vpa_alphabet = gen_arithmetic_data(num_sequances=4000, min_seq_len=2, max_seq_len=10)

    print(f"Alphabet: {vpa_alphabet}")

    print('Data:')
    for i in range(10):
        print(arithmetic_data[i])

    learned_model = run_PAPNI(arithmetic_data, vpa_alphabet)
    learned_model.visualize()


def passive_vpa_learning_on_all_benchmark_models():
    from aalpy.learning_algs import run_PAPNI
    from aalpy.utils.BenchmarkVpaModels import vpa_L1, vpa_L12, vpa_for_odd_parentheses
    from aalpy.utils import generate_input_output_data_from_vpa, convert_i_o_traces_for_RPNI

    for gt in [vpa_L1(), vpa_L12(), vpa_for_odd_parentheses()]:
        vpa_alphabet = gt.input_alphabet
        data = generate_input_output_data_from_vpa(gt, num_sequances=2000, max_seq_len=16)

        papni = run_PAPNI(data, vpa_alphabet, algorithm='gsm', print_info=True)

        for seq, o in data:
            papni.reset_to_initial()
            learned_output = papni.execute_sequence(papni.initial_state, seq)[-1]
            if o != learned_output:
                print(seq, o, learned_output)
                assert False, 'Papni Learned Model not consistent with data.'

        print('PAPNI model conforms to data.')


def gsm_rpni():
    from aalpy import load_automaton_from_file
    from aalpy.utils.Sampling import get_io_traces, sample_with_length_limits
    from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM

    automaton = load_automaton_from_file("DotModels/car_alarm.dot", "moore")
    input_traces = sample_with_length_limits(automaton.get_input_alphabet(), 100, 20, 30)
    traces = get_io_traces(automaton, input_traces)

    learned_model = run_GSM(traces, output_behavior="moore", transition_behavior="deterministic")
    learned_model.visualize()


def gsm_edsm():
    from typing import Dict
    from aalpy import load_automaton_from_file
    from aalpy.utils.Sampling import get_io_traces, sample_with_length_limits
    from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
    from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation
    from aalpy.learning_algs.general_passive.GsmNode import GsmNode

    automaton = load_automaton_from_file("DotModels/car_alarm.dot", "moore")
    input_traces = sample_with_length_limits(automaton.get_input_alphabet(), 100, 20, 30)
    traces = get_io_traces(automaton, input_traces)

    def EDSM_score(part: Dict[GsmNode, GsmNode]):
        nr_partitions = len(set(part.values()))
        nr_merged = len(part)
        return nr_merged - nr_partitions

    score = ScoreCalculation(score_function=EDSM_score)
    learned_model = run_GSM(traces, output_behavior="moore", transition_behavior="deterministic", score_calc=score)
    learned_model.visualize()


def gsm_likelihood_ratio():
    from typing import Dict
    from scipy.stats import chi2
    from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
    from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreFunction, differential_info, ScoreCalculation
    from aalpy.learning_algs.general_passive.GsmNode import GsmNode
    from aalpy.utils.Sampling import get_io_traces, sample_with_length_limits
    from aalpy import load_automaton_from_file

    automaton = load_automaton_from_file("DotModels/MDPs/faulty_car_alarm.dot", "mdp")
    input_traces = sample_with_length_limits(automaton.get_input_alphabet(), 2000, 20, 30)
    traces = get_io_traces(automaton, input_traces)

    def likelihood_ratio_score(alpha=0.05) -> ScoreFunction:
        if not 0 < alpha <= 1:
            raise ValueError(f"Confidence {alpha} not between 0 and 1")

        def score_fun(part: Dict[GsmNode, GsmNode]):
            llh_diff, param_diff = differential_info(part)
            if param_diff == 0:
                # This should cover the corner case when the partition merges only states with no outgoing transitions.
                return -1  # Let them be very bad merges.
            score = 1 - chi2.cdf(2 * llh_diff, param_diff)
            if score < alpha:
                return False
            return score

        return score_fun

    score = ScoreCalculation(score_function=likelihood_ratio_score())
    learned_model = run_GSM(traces, output_behavior="moore", transition_behavior="stochastic", score_calc=score)
    learned_model.visualize()


def gsm_IOAlergia_EDSM():
    from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
    from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import hoeffding_compatibility, ScoreCalculation
    from aalpy.learning_algs.general_passive.GsmNode import GsmNode
    from aalpy.utils.Sampling import get_io_traces, sample_with_length_limits
    from aalpy import load_automaton_from_file

    automaton = load_automaton_from_file("DotModels/MDPs/faulty_car_alarm.dot", "mdp")
    input_traces = sample_with_length_limits(automaton.get_input_alphabet(), 2000, 20, 30)
    traces = get_io_traces(automaton, input_traces)

    class IOAlergiaWithEDSM(ScoreCalculation):
        def __init__(self, epsilon):
            super().__init__()
            self.ioa_compatibility = hoeffding_compatibility(epsilon)
            self.evidence = 0

        def reset(self):
            self.evidence = 0

        def local_compatibility(self, a: GsmNode, b: GsmNode):
            self.evidence += 1
            return self.ioa_compatibility(a, b)

        def score_function(self, part: dict[GsmNode, GsmNode]):
            return self.evidence

    epsilon = 0.05
    scores = {
        "IOA": ScoreCalculation(hoeffding_compatibility(epsilon)),
        "IOA+EDSM": IOAlergiaWithEDSM(epsilon),
    }
    for name, score in scores.items():
        learned_model = run_GSM(traces, output_behavior="moore", transition_behavior="stochastic", score_calc=score,
                            compatibility_on_pta=True, compatibility_on_futures=True)
        learned_model.visualize(name)


def gsm_IOAlergia_domain_knowldege():
    from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM
    from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import hoeffding_compatibility, ScoreCalculation
    from aalpy.learning_algs.general_passive.GsmNode import GsmNode
    from aalpy.utils.Sampling import get_io_traces, sample_with_length_limits
    from aalpy import load_automaton_from_file

    automaton = load_automaton_from_file("DotModels/MDPs/faulty_car_alarm.dot", "mdp")
    input_traces = sample_with_length_limits(automaton.get_input_alphabet(), 2000, 20, 30)
    traces = get_io_traces(automaton, input_traces)

    ioa_compat = hoeffding_compatibility(0.05)

    def get_parity(node: GsmNode):
        pref = node.get_prefix()
        return [sum(in_s == key for in_s, out_s in pref) % 2 for key in ["l", "d"]]

    # The car has 4 physical states arising from the combination of locked/unlocked and open/closed.
    # Each input toggles a transition between these four states. While the car alarm system has richer behavior than that,
    # it still needs to discern the physical states. Thus, in every sane implementation of a car alarm system, every state
    # is associated with exactly one physical state. This additional assumption can be enforced by checking the parity of
    # all input symbols during merging.
    def ioa_compat_domain_knowledge(a: GsmNode, b: GsmNode):
        parity = get_parity(a) == get_parity(b)
        ioa = ioa_compat(a, b)
        return parity and ioa

    scores = {
        "IOA": ScoreCalculation(ioa_compat),
        "IOA+DK": ScoreCalculation(ioa_compat_domain_knowledge),
    }
    for name, score in scores.items():
        learned_model = run_GSM(traces, output_behavior="moore", transition_behavior="stochastic", score_calc=score,
                            compatibility_on_pta=True, compatibility_on_futures=True)
        learned_model.visualize(name)

def k_tails_example():
    from aalpy.learning_algs import run_k_tails
    from aalpy.utils import generate_random_deterministic_automata, generate_input_output_data_from_automata

    model = generate_random_deterministic_automata('moore', num_states=5,
                                                   input_alphabet_size=3,
                                                   output_alphabet_size=3)

    data = generate_input_output_data_from_automata(model, num_sequances=2000,
                                                    min_seq_len=1, max_seq_len=12,
                                                    sequance_type='io_traces')

    # k-trails works with prefix-closed input output traces, not labeled sequences like RPNI
    # data is a list of sequences in this format [(i1, o1), (i2, o1), (i1, o3)]

    # run k_tails with two different k's
    k_trails_1 = run_k_tails(data, k=3, automaton_type='moore', print_info=True)

    k_tails_2 = run_k_tails(data, k=8, automaton_type='mealy', print_info=True)
