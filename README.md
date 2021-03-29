# AALpy - An Active Automata Learning Library

AALpy is a light-weight active automata learning library written in pure Python. 
By implementing a single method and a few lines of 
configuration, you can start learning automata. 

Whether you work with regular languages or you would like to learn models of 
reactive systems, AALpy supports a wide range of modeling formalisms, including 
deterministic, non-deterministic, and stochastic automata. 
You can use it to learn **deterministic finite automata**, **Moore machines**, 
and **Mealy machines** of deterministic systems. 
If the system that you would like to learn shows non-deterministic or
stochastic behavior, AALpy allows you to learn **observable
nondeterministic finite-state machines**, **Markov decision processes**, 
or **stochastic transducers**.

AALpy enables efficient learning by providing a **large array of equivalence oracles**, implementing various **conformance testing** strategies. Learning 
is mostly based on Angluin's [L*](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf) algorithm, for which AALpy supports a 
selection of optimizations, including **efficient counterexample processing**.
Finally, support for learning **abstracted non-deterministic Mealy machines** 
enables efficient learning of system models with large input space. 

If you miss a specific feature in AALpy, you can easily extend it. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install AALpy.
```bash
pip install aalpy
```
The minimum required version of Python is 3.6.  
Ensure that you have [Graphviz](https://graphviz.org/) installed and added to your path if you want to visualize models.

For manual installation, clone the master and install the following dependency.
```bash
pip install pydot
# and to install the library
python setup.py install
```

## Documentation and Wiki

If you are interested in automata learning or would like to understand the automata learning process in more detail,
please check out our **Wiki**. On Wiki, you will find more detailed examples on how to use AALpy.
- <https://github.com/emuskardin/AALpy/wiki>

For the **official documentation** of all classes and methods, check out:
- <https://emuskardin.github.io/AALpy/documentation/index.html>

**Interactive examples** can be found in the [notebooks](https://github.com/emuskardin/AALpy/tree/master/notebooks) folder.
If you would like to interact/change those examples in the browser, click on the following badge. (Navigate to the _notebooks_ folder and select one notebook)

[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/emuskardin/AALpy/master)

[Examples.py](https://github.com/emuskardin/AALpy/blob/master/Examples.py) contains many examples demonstrating all AALpy functionality are presented. 

## Usage

All automata learning procedures follow this high-level approach:
- [Define the input alphabet and system under learning (SUL)](https://github.com/emuskardin/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems)
- [Choose the equivalence oracle](https://github.com/emuskardin/AALpy/wiki/Equivalence-Oracles)
- [Run the learning algorithm](https://github.com/emuskardin/AALpy/wiki/Setting-Up-Learning)

If you would like to learn a black-box Date Validator's behavior, your AALpy configuration would look something like this:
```python
from aalpy.base import SUL
from aalpy.utils import visualize_automaton, DateValidator
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
        # The ormat of the dates is %d/%m/%Y'
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
visualize_automaton(learned_model)
```

The following snippet demonstrates a short example in which an automaton is either [loaded](https://github.com/emuskardin/AALpy/wiki/Loading,Saving,-Syntax-and-Visualization-of-Automata) or [randomly generated](https://github.com/emuskardin/AALpy/wiki/Generation-of-Random-Automata) and then [learned](https://github.com/emuskardin/AALpy/wiki/Setting-Up-Learning).
```python
from aalpy.utils import load_automaton_from_file, save_automaton_to_file, visualize_automaton, generate_random_dfa
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_Lstar

# load an automaton
automaton = load_automaton_from_file('path_to_the_file.dot', automaton_type='dfa')

# or randomly generate one
random_dfa = generate_random_dfa(alphabet=[1,2,3,4,5],num_states=2000, num_accepting_states=200)

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
print(automaton)
```

To make experiments reproducible, define a random seed at the beginning of your program.
```Python
from random import seed
seed(2) # all experiments will be reproducible
```

An example demonstrating step-by-step instructions for learning regular expressions can be found at [How to learn Regex with AALpy](https://github.com/emuskardin/AALpy/wiki/SUL-Interface%2C-or-How-to-Learn-Your-Systems/_edit#example---regexsul).
Examples demonstrating how to learn Bluetooth or other IoT protocols like MQTT can be found at [How to learn MQTT with AALpy](https://github.com/emuskardin/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems#example---mqtt). 

For more examples and instructions check out the [Wiki](https://github.com/emuskardin/AALpy/wiki
) , [notebooks](https://github.com/emuskardin/AALpy/tree/master/notebooks), and [Examples.py](https://github.com/emuskardin/AALpy/blob/master/Examples.py).

## Cite AALpy
If you use AALpy in your research, please cite:
```
To be announced, please contact edi.muskardin@silicon-austria.com in the meantime. 
```

## Research Contact
If you have research suggestions or specific help with respect to your research, wheter it is automata learning itself or application
of automata learning, feel free to contact [edi.muskardin@silicon-austria.com](mailto:edi.muskardin@silicon-austria.com).
We are happy to help you and consult you in applyingg automata learning in various domain.

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.
In case of any questions or possible bugs, please open issues.

## Contributors
- Edi Muskardin
- Martin Tappler

## License
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
