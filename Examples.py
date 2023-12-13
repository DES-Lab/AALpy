def random_deterministic_model_example():
    from aalpy.utils import generate_random_deterministic_automata
    from aalpy.SULs import MealySUL
    from aalpy.oracles import RandomWMethodEqOracle
    from aalpy.learning_algs import run_KV

    model_type = 'mealy'  # or 'moore', 'dfa'

    # for random dfa's you can also define num_accepting_states
    random_model = generate_random_deterministic_automata(automaton_type=model_type, num_states=100,
                                                          input_alphabet_size=3, output_alphabet_size=4)

    sul = MealySUL(random_model)
    input_alphabet = random_model.get_input_alphabet()

    # select any of the oracles
    eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=10, walk_len=20)

    learned_model = run_KV(input_alphabet, sul, eq_oracle, model_type)

    return learned_model


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


def random_deterministic_example_with_provided_sequences():
    from random import choice, randint
    from aalpy.SULs import MealySUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.utils import generate_random_deterministic_automata

    random_mealy = generate_random_deterministic_automata('mealy', num_states=10,
                                                          input_alphabet_size=4, output_alphabet_size=3)

    input_alphabet = random_mealy.get_input_alphabet()

    sul_mealy = MealySUL(random_mealy)

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


def big_input_alphabet_example(input_alphabet_size=1000, automaton_depth=4):
    """
        Small example where input alphabet can be huge and outputs are just true and false (DFA).

    Args:
        input_alphabet_size: size of input alphabet
        automaton_depth: depth of alternating True/False paths in final automaton

    Returns:
        learned model
    """
    from aalpy.base import SUL
    from aalpy.learning_algs import run_Lstar
    from aalpy.oracles import RandomWMethodEqOracle

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
    from aalpy.SULs import OnfsmSUL
    from aalpy.utils import generate_random_ONFSM
    from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
    from aalpy.learning_algs import run_non_det_Lstar

    onfsm = generate_random_ONFSM(num_states=num_states, num_inputs=input_size, num_outputs=output_size)
    alphabet = onfsm.get_input_alphabet()

    sul = OnfsmSUL(onfsm)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=500, reset_prob=0.15, reset_after_cex=True)
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=8, max_walk_len=20)

    learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle=eq_oracle, n_sampling=n_sampling, print_level=2)
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
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import generate_random_mdp

    mdp = generate_random_mdp(num_states, input_len, num_outputs)
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)
    eq_oracle = RandomWalkEqOracle(input_alphabet, sul=sul, num_steps=5000, reset_prob=0.11,
                                   reset_after_cex=False)

    learned_mdp = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, n_c=n_c, n_resample=n_resample,
                                       min_rounds=min_rounds, max_rounds=max_rounds)

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

    from aalpy.SULs import OnfsmSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_non_det_Lstar
    from aalpy.utils import get_benchmark_ONFSM

    onfsm = get_benchmark_ONFSM()
    alphabet = onfsm.get_input_alphabet()

    sul = OnfsmSUL(onfsm)
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
    from aalpy.SULs import MealySUL
    from aalpy.utils import load_automaton_from_file

    class Multi_Client_MQTT_Mapper(SUL):
        def __init__(self):
            super().__init__()

            five_clients_mqtt_mealy = load_automaton_from_file('DotModels/five_clients_mqtt_abstracted_onfsm.dot',
                                                               automaton_type='mealy')
            self.five_client_mqtt = MealySUL(five_clients_mqtt_mealy)
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
    from aalpy.SULs import OnfsmSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_abstracted_ONFSM_Lstar
    from aalpy.utils import get_ONFSM

    onfsm = get_ONFSM()

    alphabet = onfsm.get_input_alphabet()

    sul = OnfsmSUL(onfsm)
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
    from aalpy.SULs import MdpSUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import get_faulty_coffee_machine_MDP

    mdp = get_faulty_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)

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
    from aalpy.SULs import MdpSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import get_weird_coffee_machine_MDP

    mdp = get_weird_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)

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
    from aalpy.SULs import MdpSUL
    from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar
    from aalpy.utils import load_automaton_from_file

    # Specify the path to the dot file containing a MDP
    mdp = load_automaton_from_file(f'./DotModels/MDPs/{example}.dot', automaton_type='mdp')
    input_alphabet = mdp.get_input_alphabet()

    sul = MdpSUL(mdp)
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
    from aalpy.SULs import MdpSUL, StochasticMealySUL
    from aalpy.automata import Mdp
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_stochastic_Lstar

    input_al = stochastic_machine.get_input_alphabet()

    if isinstance(stochastic_machine, Mdp):
        sul = MdpSUL(stochastic_machine)
    else:
        sul = StochasticMealySUL(stochastic_machine)

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
    from aalpy.SULs import MdpSUL
    from random import randint, choice
    from aalpy.learning_algs import run_Alergia
    from aalpy.utils import generate_random_mdp

    mdp = generate_random_mdp(5, 2, 3)
    sul = MdpSUL(mdp)
    inputs = mdp.get_input_alphabet()

    data = []
    for _ in range(100000):
        str_len = randint(5, 12)
        # add the initial output
        seq = [sul.pre()]
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
    from aalpy.SULs import StochasticMealySUL
    from random import randint, choice
    from aalpy.learning_algs import run_Alergia
    from aalpy.utils import generate_random_smm

    smm = generate_random_smm(5, 2, 2)
    sul = StochasticMealySUL(smm)
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


def alergia_mc_example():
    from os import remove
    from aalpy.SULs import McSUL
    from random import randint
    from aalpy.learning_algs import run_Alergia
    from aalpy.utils import generate_random_markov_chain
    from aalpy.utils import CharacterTokenizer

    mc = generate_random_markov_chain(10)
    mc.visualize('Original')

    sul = McSUL(mc)

    # note that this example shows writing to file just to show how tokenizer is used...
    # this step can ofc be skipped and lists passed to alergia
    data = []
    for _ in range(20000):
        str_len = randint(4, 12)
        seq = [f'{sul.pre()}']
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
    from aalpy.SULs import MdpSUL
    from aalpy.utils import load_automaton_from_file
    from aalpy.learning_algs import run_active_Alergia
    from aalpy.learning_algs.stochastic_passive.ActiveAleriga import RandomWordSampler

    mdp = load_automaton_from_file(f'./DotModels/MDPs/{example}.dot', automaton_type='mdp')
    input_alphabet = mdp.get_input_alphabet()

    sul = MdpSUL(mdp)

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
    import random
    from aalpy.SULs import MooreSUL
    from aalpy.learning_algs import run_RPNI
    from aalpy.oracles import StatePrefixEqOracle
    from aalpy.utils import generate_random_moore_machine, generate_random_dfa

    model = generate_random_dfa(num_states=5, alphabet=[1, 2, 3], num_accepting_states=2)
    model = generate_random_moore_machine(num_states=5, input_alphabet=[1, 2, 3], output_alphabet=['a', 'b'])

    input_al = model.get_input_alphabet()

    num_sequences = 1000
    data = []
    for _ in range(num_sequences):
        seq_len = random.randint(1, 20)
        random_seq = random.choices(input_al, k=seq_len)
        output = model.compute_output_seq(model.initial_state, random_seq)[-1]
        data.append((random_seq, output))

    rpni_model = run_RPNI(data, automaton_type='moore', print_info=True)

    rpni_model.make_input_complete('sink_state')
    sul = MooreSUL(model)
    eq_oracle_2 = StatePrefixEqOracle(input_al, sul, walks_per_state=100)
    cex = eq_oracle_2.find_cex(rpni_model)
    if cex is None:
        print("Could not find a counterexample between the RPNI-model and the original model.")
    else:
        print('Counterexample found. Either RPNI data was incomplete, or there is a bug in RPNI algorithm :o ')


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
    from aalpy.SULs import MealySUL

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
    sul = MealySUL(model)
    active_rpni_model = run_active_RPNI(data, sul, sampler=sampler, n_iter=5,
                                        automaton_type='mealy', print_info=True)

    return active_rpni_model


def compare_stochastic_and_non_deterministic_learning(example='first_grid'):
    import aalpy.paths
    aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
    aalpy.paths.path_to_properties = "Benchmarking/prism_eval_props/"

    from aalpy.SULs import MdpSUL, OnfsmSUL
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
    sul = MdpSUL(mdp)
    eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=100, min_walk_len=5, max_walk_len=15,
                                   reset_after_cex=True)
    stochastic_learned_model = run_stochastic_Lstar(input_alphabet=input_alphabet, eq_oracle=eq_oracle, sul=sul,
                                                    automaton_type='smm', target_unambiguity=0.99)

    # Non Deterministic Learning
    print("Non-deterministic Learning")
    sul = OnfsmSUL(mdp)
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
                self.string_under_test += ' ' + letter

            try:
                eval(self.string_under_test)
                return True
            except (SyntaxError, TypeError):
                return False

    sul = ArithmeticSUL()

    alphabet = SevpaAlphabet(internal_alphabet=['1', '+'], call_alphabet=['('], return_alphabet=[')'])

    eq_oracle = RandomWordEqOracle(alphabet.get_merged_alphabet(), sul, min_walk_len=5,
                                   max_walk_len=20, num_walks=20000)

    learned_model = run_KV(alphabet, sul, eq_oracle, automaton_type='vpa')
    learned_model.visualize()


def benchmark_sevpa_learning():
    from aalpy.SULs import SevpaSUL
    from aalpy.oracles import RandomWordEqOracle
    from aalpy.learning_algs import run_KV
    from aalpy.utils.BenchmarkSevpaModels import sevpa_for_L1, sevpa_for_L2, sevpa_for_L11, sevpa_for_L12, sevpa_for_L14

    models = [sevpa_for_L1(), sevpa_for_L2(), sevpa_for_L11(), sevpa_for_L12(), sevpa_for_L14()]

    for inx, model in enumerate(models):

        alphabet = model.get_input_alphabet()

        sul = SevpaSUL(model)

        if inx == 4:
            alphabet.exclusive_call_return_pairs = {'(': ')', '[': ']'}

        eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000,
                                       min_walk_len=10, max_walk_len=30)

        learned_model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                               print_level=2, cex_processing='rs')

        print(learned_model.get_random_accepting_word())


def random_sevpa_learning():
    from aalpy.SULs import SevpaSUL
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

    sul = SevpaSUL(random_svepa)

    eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000,
                                   min_walk_len=10, max_walk_len=30)

    model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                   print_level=2, cex_processing='rs')

