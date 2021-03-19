# AALpy - An Active Automata Learning Library
<!-- ![Alt text](saly.png?raw=true "AALpy") -->

AALpy is light-weight a active automata learning library written in pure Python. 
By implementing a single method and a few lines of 
configuration, you can start learning automata. 

Whether you work with regular languages, or you want to learn models of 
reactive systems, AALpy supports a wide range of modelling formalisms including 
deterministic, non-deterministic, and stochastic automata. 
You can use it to learn **deterministic finite automata**, **Moore machines**, 
and **Mealy machines** of deterministic systems. 
If the system that you want to learn shows non-deterministic or
stochastic behavior, you can use AALpy to learn **observable
nondeterministic finite-state machines**, **Markov decision processes**, 
or **stochastic transducers**.

AALpy enables efficient learning by providing a **large array of equivalence 
oracles**, implementing various **conformance testing** strategies. Learning 
is mostly based on Angluin's L* algorithm, for which AALpy supports a 
selection of optimizations, including **efficient counterexample processing**.
Finally, support for learning **abstracted nondeterministic Mealy machines** 
enables efficient learning of system models with large input space. 

If AALpy misses a feature that you need, you can easily extend it. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install AALpy.
```bash
pip install aalpy
```
Ensure that you have [Graphviz](https://graphviz.org/) installed and added to your path if you want to visualize models.

For manual installation clone the master and install the following dependency.
```bash
pip install pydot
```

## Documentation and Wiki

If you are interested in automata learning or want to understand the automata learning process in more detail,
please check out our Wiki. On Wiki, you will find more detailed examples on how to use ALLpy.
- https://github.com/EdiMuskardin/AALpy/wiki

For the official documentation of all classes and methods check out:
- https://edimuskardin.github.io/AALpy/

Furthermore, there you will find everything from an introduction to automata learning to the discussion of some advanced topics.

In [Examples.py](https://github.com/EdiMuskardin/AALpy/blob/master/Examples.py), many examples demonstrating all ALLpy functionality are presented. 


## Usage

All automata learning procedures follow this high-level approach:
- Define input alphabet
- Define system under learning (SUL)
- Define equivalence oracle
- Run learning algorithm with input alphabet

The following snippet demonstrates a short example in which automaton is either loaded or randomly generated and then learned.
```python
from aalpy.utils import load_automaton_from_file, save_automaton_to_file, visualize_automaton, generate_random_dfa
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle, StatePrefixEqOracle
from aalpy.learning_algs import run_Lstar

# load an automaton
automaton = load_automaton_from_file('path_to_the_file.dot')
# or randomly generate one
random_dfa = generate_random_dfa(alphabet=[1,2,3,4,5],num_states=2000, num_accepting_states=200)
# get input alphabet of the automaton
alphabet = random_dfa.get_input_alphabet()
# create a SUL instance for the automaton/system under learning
sul = DfaSUL(automaton)

# define the equivalence oracle
eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09)
eq_oracle_2 = StatePrefixEqOracle(alphabet, sul, walks_per_state=20, walk_len=10)

# start learning
learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa')

# save automaton to file and visualize it
save_automaton_to_file(learned_dfa, path='Learned_Automaton', file_type='dot')
visualize_automaton(learned_dfa)
```

### Input alphabet
Input alphabet is a list of any hashable object, most often strings or integers.
In the step method of the SUL interface receives an element from input alphabet and
performs a system action based on it.

From standard automatons (loaded from file/randomly generated), input alphabet can be obtained with:
```python
automaton.get_input_alphabet()
```

To see how class methods and their arguments can be used as the elements of the input alphabet, check out
[aalpy/SULs/PyMethodSUL](https://github.com/EdiMuskardin/AALpy/blob/master/aalpy/SULs/PyMethodSUL.py)
 and learnPythonClass in [Examples.py](https://github.com/EdiMuskardin/AALpy/blob/master/Examples.py).

### SUL Interface

The system under learning (SUL) is a basic abstract class that all systems that you want to learn have to implement.
When inferring loaded or randomly generated automata, one only has to pass it to the appropriate SUL found in 
[aalpy/SULs/AutomataSUL.py](https://github.com/EdiMuskardin/AALpy/blob/master/aalpy/SULs/AutomataSUL.py).
However, if you want to learn any other system, you have to implement a system reset and step method.

Reset is divided in
- pre() : initialization/startup of the system
- post() : gracefull shutdown of the system/memory cleanup

The step method received an element of the input alphabet.
Based on the received element certain action is performed.
Membership query is a sequence of steps.

For further examples on how to implement SUL refer to:
[aalpy/SULs/](https://github.com/EdiMuskardin/AALpy/blob/master/aalpy/SULs/)

### Loading and saving automatas to file
```python
from aalpy.utils import save_automaton_to_file, load_automaton_from_file, visualize_automaton

# loading automaton from file
dfa = load_automaton_from_file(path='path_to_the_file', automaton_type='dfa')
# if compute prefixes is set to true, prefix/shortest path till each state will be computed
mealy = load_automaton_from_file(path='path_to_mealy', automaton_type='mealy', compute_prefixes=True)
mdp = load_automaton_from_file(path='path_to_the_file', automaton_type='mdp')

# write dot representation of the automaton to file
save_automaton_to_file(mdp, path='new_mdp.dot')

# visualize automaton
# supported formats are 'png', 'svg', 'pdf'
visualize_automaton(dfa, file_type='pdf')

# print dot representation of the automaton to standard output
print(dfa)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
In case of any questions or possible bugs, please open issues.

## License
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
