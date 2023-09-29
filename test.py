from random import seed

from aalpy.SULs import MdpSUL
from random import randint, choice
from aalpy.learning_algs import run_Alergia
from aalpy.utils import generate_random_mdp

seed(1)

mdp = generate_random_mdp(20, 2, 3)
sul = MdpSUL(mdp)
inputs = mdp.get_input_alphabet()

data = []

for _ in range(10000):
    str_len = randint(5, 20)
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
