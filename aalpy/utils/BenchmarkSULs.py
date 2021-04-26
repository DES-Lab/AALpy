from aalpy.automata import StochasticMealyMachine, StochasticMealyState, Mdp, MdpState, Onfsm, OnfsmState, \
    DfaState, Dfa


def get_Angluin_dfa():
    q0 = DfaState('q0')
    q0.is_accepting = True
    q1 = DfaState('q1')
    q2 = DfaState('q2')
    q3 = DfaState('q3')

    q0.transitions['a'] = q1
    q0.transitions['b'] = q2
    q1.transitions['a'] = q0
    q1.transitions['b'] = q3
    q2.transitions['a'] = q3
    q2.transitions['b'] = q0
    q3.transitions['a'] = q2
    q3.transitions['b'] = q1

    return Dfa(q0, [q0, q1, q2, q3])


def get_benchmark_ONFSM():
    """
    Returns ONFSM presented in 'Learning Finite State Models of Observable Nondeterministic Systems in a Testing
    Context'.
    """
    a = OnfsmState('a')
    b = OnfsmState('b')
    c = OnfsmState('c')
    d = OnfsmState('d')

    a.transitions['a'].append((0, b))
    a.transitions['b'].append((2, a))
    a.transitions['b'].append((0, c))

    b.transitions['a'].append((2, a))
    b.transitions['b'].append((3, b))

    c.transitions['a'].append((2, d))
    c.transitions['b'].append((0, c))
    c.transitions['b'].append((3, c))

    d.transitions['a'].append((2, b))
    d.transitions['b'].append((3, d))

    return Onfsm(a, [a, b, c, d])


def get_ONFSM():
    """
    Returns example of an ONFSM.
    """
    q0 = OnfsmState('q0')
    q1 = OnfsmState('q1')
    q2 = OnfsmState('q2')
    q3 = OnfsmState('q3')
    q4 = OnfsmState('q4')
    q5 = OnfsmState('q5')
    q6 = OnfsmState('q6')
    q7 = OnfsmState('q7')
    q8 = OnfsmState('q8')

    q0.transitions['a'].append((2, q1))
    q0.transitions['b'].append((0, q0))

    q1.transitions['a'].append((2, q0))
    q1.transitions['b'].append((0, q2))

    q2.transitions['a'].append((1, q2))
    q2.transitions['b'].append((0, q3))

    q3.transitions['a'].append((2, q8))
    q3.transitions['b'].append((0, q4))

    q4.transitions['a'].append((1, q4))
    q4.transitions['b'].append((0, q5))

    q5.transitions['a'].append((2, q6))
    q5.transitions['b'].append((0, q7))

    q6.transitions['a'].append((2, q5))
    q6.transitions['b'].append((0, q6))

    q7.transitions['a'].append((1, q7))
    q7.transitions['b'].append(('O', q0))

    q8.transitions['a'].append((2, q3))
    q8.transitions['b'].append((0, q8))

    return Onfsm(q0, [q0, q1, q2, q3, q4, q5, q6, q7, q8])


def get_faulty_coffee_machine_MDP():
    q0 = MdpState("q0", "init")
    q1 = MdpState("q1", "beep")
    q2 = MdpState("q2", "coffee")

    q0.transitions['but'].append((q0, 1))
    q0.transitions['coin'].append((q1, 1))
    q1.transitions['but'].append((q0, 0.1))
    q1.transitions['but'].append((q2, 0.9))
    q1.transitions['coin'].append((q1, 1))
    q2.transitions['but'].append((q0, 1))
    q2.transitions['coin'].append((q1, 1))

    mdp = Mdp(q0, [q0, q1, q2])

    return mdp


def get_weird_coffee_machine_MDP():
    q0 = MdpState("q0", "init")
    q1 = MdpState("q1", "beep")
    q2 = MdpState("q2", "coffee")
    q3 = MdpState("q3", "beep")
    q4 = MdpState("q4", "coffee")
    q5 = MdpState("q5", "init")
    q6 = MdpState("q6", "crash")

    q0.transitions['but'].append((q0, 1))
    q0.transitions['coin'].append((q1, 1))
    q0.transitions['koin'].append((q3, 1))

    q1.transitions['but'].append((q0, 0.1))
    q1.transitions['but'].append((q2, 0.9))

    q3.transitions['but'].append((q0, 0.1))
    q3.transitions['but'].append((q4, 0.9))

    q1.transitions['coin'].append((q1, 1))
    q3.transitions['koin'].append((q3, 1))
    q1.transitions['koin'].append((q3, 1))
    q3.transitions['coin'].append((q1, 1))

    q2.transitions['but'].append((q0, 1))
    q2.transitions['coin'].append((q1, 1))
    q2.transitions['koin'].append((q3, 1))

    q4.transitions['coin'].append((q1, 1))
    q4.transitions['koin'].append((q3, 1))

    q4.transitions['but'].append((q5, 1))

    q5.transitions['but'].append((q6, 1))
    q5.transitions['coin'].append((q6, 1))
    q5.transitions['koin'].append((q6, 1))

    q6.transitions['but'].append((q6, 1))
    q6.transitions['coin'].append((q6, 1))
    q6.transitions['koin'].append((q6, 1))

    mdp = Mdp(q0, [q0, q1, q2, q3, q4, q5, q6])

    return mdp


def get_faulty_coffee_machine_SMM():
    s0 = StochasticMealyState('s0')
    s1 = StochasticMealyState('s1')
    s2 = StochasticMealyState('s2')

    s0.transitions['but'].append((s0, 'init', 1.))
    s0.transitions['coin'].append((s1, 'beep', 1.))
    s1.transitions['but'].append((s0, 'init', 0.1))
    s1.transitions['but'].append((s2, 'coffee', 0.9))
    s1.transitions['coin'].append((s1, 'beep', 1.))
    s2.transitions['but'].append((s0, 'init', 1.))
    s2.transitions['coin'].append((s1, 'beep', 1.))

    smm = StochasticMealyMachine(s0, [s0, s1, s2])

    return smm


def get_minimal_faulty_coffee_machine_SMM():
    s0 = StochasticMealyState('s0')
    s1 = StochasticMealyState('s1')

    s0.transitions['but'].append((s0, 'init', 1.))
    s0.transitions['coin'].append((s1, 'beep', 1.))
    s1.transitions['but'].append((s0, 'init', 0.1))
    s1.transitions['but'].append((s0, 'coffee', 0.9))
    s1.transitions['coin'].append((s1, 'beep', 1.))

    smm = StochasticMealyMachine(s0, [s0, s1])

    return smm


def get_faulty_mqtt_SMM():
    s0 = StochasticMealyState('s0')
    s1 = StochasticMealyState('s1')
    s2 = StochasticMealyState('s2')

    s0.transitions['connect'].append((s1, 'CONNACK', 1.))
    s0.transitions['disconnect'].append((s0, 'CONCLOSED', 1.))
    s0.transitions['publish'].append((s0, 'CONCLOSED', 1.))
    s0.transitions['subscribe'].append((s0, 'CONCLOSED', 1.))
    s0.transitions['unsubscribe'].append((s0, 'CONCLOSED', 1.))

    s1.transitions['connect'].append((s0, 'CONCLOSED', 1.))
    s1.transitions['disconnect'].append((s0, 'CONCLOSED', 1.))
    s1.transitions['publish'].append((s1, 'PUBACK', 0.9))
    s1.transitions['publish'].append((s0, 'CONCLOSED', 0.1))
    s1.transitions['subscribe'].append((s2, 'SUBACK', 1.))
    s1.transitions['unsubscribe'].append((s1, 'UNSUBACK', 1.))

    s2.transitions['connect'].append((s0, 'CONCLOSED', 1.))
    s2.transitions['disconnect'].append((s0, 'CONCLOSED', 1.))
    s2.transitions['publish'].append((s2, 'PUBLISH_PUBACK', 1.))
    s2.transitions['subscribe'].append((s2, 'SUBACK', 1.))
    s2.transitions['unsubscribe'].append((s1, 'UNSUBACK', 0.8))
    s2.transitions['unsubscribe'].append((s2, 'SUBACK', 0.2))

    smm = StochasticMealyMachine(s0, [s0, s1, s2])

    return smm


def get_small_gridworld():
    s0 = StochasticMealyState('s0')
    s1 = StochasticMealyState('s1')
    s2 = StochasticMealyState('s2')
    s3 = StochasticMealyState('s3')

    p_g = 0.8
    p_m = 0.6

    # gridworld of the form
    # W W W W with a start in the top left
    # W G M W states like s0 s1
    # W M G W             s2 s3
    # W W W W

    s0.transitions['north'].append((s0, 'wall', 1.))
    s0.transitions['west'].append((s0, 'wall', 1.))
    s0.transitions['east'].append((s1, 'mud', p_m))
    s0.transitions['east'].append((s3, 'grass', 1 - p_m))
    s0.transitions['south'].append((s2, 'mud', p_m))
    s0.transitions['south'].append((s3, 'grass', 1 - p_m))

    s1.transitions['north'].append((s1, 'wall', 1.))
    s1.transitions['east'].append((s1, 'wall', 1.))
    s1.transitions['west'].append((s0, 'grass', p_g))
    s1.transitions['west'].append((s2, 'mud', 1 - p_g))
    s1.transitions['south'].append((s3, 'grass', p_g))
    s1.transitions['south'].append((s2, 'mud', 1 - p_g))

    s2.transitions['south'].append((s2, 'wall', 1.))
    s2.transitions['west'].append((s2, 'wall', 1.))
    s2.transitions['east'].append((s3, 'grass', p_g))
    s2.transitions['east'].append((s1, 'mud', 1 - p_g))
    s2.transitions['north'].append((s0, 'grass', p_g))
    s2.transitions['south'].append((s1, 'mud', 1 - p_g))

    s3.transitions['south'].append((s3, 'wall', 1.))
    s3.transitions['east'].append((s3, 'wall', 1.))
    s3.transitions['west'].append((s2, 'mud', p_m))
    s3.transitions['west'].append((s0, 'grass', 1 - p_m))
    s3.transitions['north'].append((s1, 'mud', p_m))
    s3.transitions['north'].append((s0, 'grass', 1 - p_m))

    smm = StochasticMealyMachine(s0, [s0, s1, s2, s3])

    return smm


class MockMqttExample:

    def __init__(self):
        self.state = 'CONCLOSED'
        self.topics = set()

    def subscribe(self, topic: str):
        if '\n' in topic or '\u0000' in topic:
            self.state = 'CONCLOSED'
            self.topics.clear()
        elif self.state != 'CONCLOSED':
            self.topics.add(topic)
            self.state = 'SUBACK'

        return self.state

    def unsubscribe(self, topic):
        if '\n' in topic or '\u0000' in topic:
            self.state = 'CONCLOSED'
            self.topics.clear()
        elif self.state != 'CONCLOSED':
            if topic in self.topics:
                self.topics.remove(topic)
            self.state = 'UNSUBACK'

        return self.state

    def connect(self):
        if self.state == 'CONCLOSED':
            self.state = 'CONNACK'
        else:
            self.topics.clear()
            self.state = 'CONCLOSED'
        return self.state

    def disconnect(self):
        self.state = 'CONCLOSED'
        self.topics.clear()
        return self.state

    def publish(self, topic):
        if '\n' in topic or '\u0000' in topic:
            self.state = 'CONCLOSED'
            self.topics.clear()
        if self.state != 'CONCLOSED':
            if topic not in self.topics:
                self.state = 'PUBACK'
            else:
                self.state = 'PUBACK_PUBACK'
        return self.state


class DateValidator:
    """
    Class mimicking Date Validator API.
    It does not account for the leap years.
    The format of the dates is %d/%m/%Y'
    """

    def is_date_accepted(self, date_string: str):
        values = date_string.split('/')
        if len(values) != 3:
            return False
        try:
            day = int(values[0])
            month = int(values[1])
            year = int(values[2])
        except ValueError:
            return False

        if not (0 <= year <= 9999):
            return False

        if month == 2 and not (1 <= day <= 28):
            return False

        if month in [1, 3, 5, 7, 8, 10, 12] and not (1 <= day <= 31):
            return False

        elif not (1 <= day <= 31):
            return False

        return True
