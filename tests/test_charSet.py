import unittest

from aalpy.utils import get_Angluin_dfa, load_automaton_from_file
from aalpy.utils.HelperFunctions import all_suffixes


class TestCharSet(unittest.TestCase):

    def get_test_automata(self):
        return {"angluin_dfa": get_Angluin_dfa(),
                "angluin_mealy": load_automaton_from_file('../DotModels/Angluin_Mealy.dot', automaton_type='mealy'),
                "angluin_moore": load_automaton_from_file('../DotModels/Angluin_Moore.dot', automaton_type='moore'),
                "mqtt": load_automaton_from_file('../DotModels/MQTT/emqtt__two_client_will_retain.dot',
                                                 automaton_type='mealy'),
                "openssl": load_automaton_from_file('../DotModels/TLS/OpenSSL_1.0.2_server_regular.dot',
                                                    automaton_type='mealy'),
                "tcp_server": load_automaton_from_file('../DotModels/TCP/TCP_Linux_Client.dot',
                                                       automaton_type='mealy')}

    def test_can_differentiate(self):
        automata = self.get_test_automata()
        for init_with_alphabet in [True, False]:
            for (online_suffix_closure, split_all_blocks) in [(False, False), (False, True), (True, False),
                                                              (True, True)]:
                for test_aut_name in automata:
                    print(f"Testing with {test_aut_name}")
                    test_aut = automata[test_aut_name]
                    char_set_init = list(map(lambda input: tuple([input]), test_aut.get_input_alphabet())) \
                        if init_with_alphabet else None
                    if "dfa" in test_aut_name or "moore" in test_aut_name:
                        char_set_init = [] if char_set_init is None else char_set_init
                        char_set_init.append(())
                    char_set = test_aut.compute_characterization_set(char_set_init=char_set_init,
                                                                     online_suffix_closure=online_suffix_closure,
                                                                     split_all_blocks=split_all_blocks)
                    print(f"Char. set {char_set}")
                    all_responses = set()
                    for s in test_aut.states:
                        responses_from_s = []
                        for c in char_set:
                            responses_from_s.append(tuple(test_aut.compute_output_seq(s, c)))
                        all_responses.add(tuple(responses_from_s))

                    # every state must have a unique response to the whole characterization set
                    assert len(all_responses) == len(test_aut.states)

    def test_suffix_closed(self):
        automata = self.get_test_automata()
        for init_with_alphabet in [True, False]:
            online_suffix_closure = True
            for split_all_blocks in [True, False]:
                for test_aut_name in automata:
                    print(f"Testing with {test_aut_name}")
                    test_aut = automata[test_aut_name]
                    char_set_init = list(map(lambda input: tuple([input]), test_aut.get_input_alphabet())) \
                        if init_with_alphabet else None
                    if "dfa" in test_aut_name or "moore" in test_aut_name:
                        char_set_init = [] if char_set_init is None else char_set_init
                        char_set_init.append(())
                    char_set = test_aut.compute_characterization_set(char_set_init=char_set_init,
                                                                     online_suffix_closure=online_suffix_closure,
                                                                     split_all_blocks=split_all_blocks)
                    print(f"Char. set {char_set}")
                    for s in char_set:
                        for suffix in all_suffixes(s):
                            if suffix not in char_set:
                                print(suffix)
                            assert suffix in char_set
