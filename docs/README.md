# Dive into Automata Learning with AALpy

Whether you work with regular languages, or you would like to learn models of 
reactive systems, AALpy supports a wide range of modeling formalisms including 
deterministic, non-deterministic, and stochastic automata. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install AALpy.
```bash
pip install aalpy
```
Ensure that you have [Graphviz](https://graphviz.org/) installed and added to your path if you want to visualize models.

## Documentation and Wiki

Please check out our **Wiki**. On Wiki, you will find more detailed examples on how to use AALpy.
- [https://github.com/emuskardin/AALpy/wiki](https://github.com/emuskardin/AALpy/wiki)

For the **official documentation** of all classes and methods check out:
- [https://emuskardin.github.io/AALpy/docs_index.html](https://emuskardin.github.io/AALpy/docs_index.html)

**Interactive examples** can be found in the [notebooks](https://github.com/emuskardin/AALpy/tree/master/notebooks) folder.
If you would like to interact/change those examples in the browser, click on the following badge. (Navigate to the _notebooks_ folder and select one notebook)

[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/emuskardin/AALpy/master)

[Examples.py](https://github.com/emuskardin/AALpy/blob/master/Examples.py) contains many examples demonstrating all AALpy functionality are presented. 

## Usage

All automata learning procedures follow this high-level approach:
- [Define the input alphabet and system under learning (SUL)](https://github.com/emuskardin/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems)
- [Choose the equivalence oracle](https://github.com/emuskardin/AALpy/wiki/Equivalence-Oracles)
- [Run the learning algorithm](https://github.com/emuskardin/AALpy/wiki/Setting-Up-Learning)

The following snippet demonstrates a short example in which an automaton is either [loaded](https://github.com/emuskardin/AALpy/wiki/Loading,Saving,-Syntax-and-Visualization-of-Automata) or [randomly generated](https://github.com/emuskardin/AALpy/wiki/Generation-of-Random-Automata) and then [learned](https://github.com/emuskardin/AALpy/wiki/Setting-Up-Learning).
```python
from aalpy.utils import save_automaton_to_file, visualize_automaton, generate_random_dfa
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_Lstar

# or randomly generate one
alphabet=[1,2,3,4,5]
random_dfa = generate_random_dfa(alphabet=alphabet, num_states=2000, num_accepting_states=200)

# get input alphabet of the automaton
alphabet = random_dfa.get_input_alphabet()

# create a SUL instance for the automaton/system under learning
sul = DfaSUL(random_dfa)

# define the equivalence oracle
eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09)

# start learning
learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa')

# save automaton to file and visualize it
save_automaton_to_file(learned_dfa, path='Learned_Automaton', file_type='dot')
# visualize automaton
visualize_automaton(learned_dfa)
# or just print its DOT representation
print(learned_dfa)
```

In order to make experiments reproducible, simply define a random seed at the beginning of your program.
```python
from random import seed
seed(2) # all experiments will be reproducible
```

An example demonstrating step-by-step instructions for learning regular expressions can be found at [How to learn Regex with AALpy](https://github.com/emuskardin/AALpy/wiki/SUL-Interface%2C-or-How-to-Learn-Your-Systems/_edit#example---regexsul).
