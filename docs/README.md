# Dive into Automata Learning with AALpy

Whether you work with regular languages or you would like to learn models of 
reactive systems, AALpy supports a wide range of modeling formalisms, including 
**deterministic**, **non-deterministic**, and **stochastic automata**. 

AALpy enables efficient learning by providing a **large array of equivalence oracles**, implementing various **conformance testing** strategies. 

Learning is mostly based on Angluin's [L* algorithm](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf), for which AALpy supports a 
selection of optimizations, including **efficient counterexample processing** and **query caching**.
Finally, support for learning **abstracted nondeterministic Mealy machines** 
enables efficient learning of system models with large input space. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install AALpy.
```bash
pip install aalpy
```
Ensure that you have [Graphviz](https://graphviz.org/) installed and added to your path if you want to visualize models.

## Documentation and Wiki

Please check out our **Wiki**. On Wiki, you will find more detailed examples on how to use AALpy.
- <https://github.com/DES-Lab/AALpy/wiki>

For the **official documentation** of all classes and methods, check out:
- <https://des-lab.github.io/AALpy/documentation/index.html>

**Interactive examples** can be found in the [notebooks](https://github.com/DES-Lab/AALpy/tree/master/notebooks) folder.
If you would like to interact/change those examples in the browser, click on the following badge. (Navigate to the _notebooks_ folder and select one notebook)

[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/DES-Lab/AALpy/master)

[Examples.py](https://github.com/DES-Lab/AALpy/blob/master/Examples.py) contains many examples demonstrating all AALpy functionality are presented. 

## Usage

All automata learning procedures follow this high-level approach:
- [Define the input alphabet and system under learning (SUL)](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems)
- [Choose the equivalence oracle](https://github.com/DES-Lab/AALpy/wiki/Equivalence-Oracles)
- [Run the learning algorithm](https://github.com/DES-Lab/AALpy/wiki/Setting-Up-Learning)

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
visualize_automaton(learned_model)
```

To make experiments reproducible, define a random seed at the beginning of your program.
```python
from random import seed
seed(2) # all experiments will be reproducible
```

Automatons can be [loaded, saved or visualized](https://github.com/DES-Lab/AALpy/wiki/Loading,Saving,-Syntax-and-Visualization-of-Automata) or [randomly generated](https://github.com/DES-Lab/AALpy/wiki/Generation-of-Random-Automata).

For more detailed examples, check out:
- [How to learn Regex with AALpy](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface%2C-or-How-to-Learn-Your-Systems/_edit#example---regexsul)
- [How to learn MQTT with AALpy](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems#example---mqtt)
- [Interactive Examples](https://github.com/DES-Lab/AALpy/tree/master/notebooks)
- [Examples.py](https://github.com/DES-Lab/AALpy/blob/master/Examples.py)

## Research Contact
If you have research suggestions or need specific help concerning your research, feel free to contact [edi.muskardin@silicon-austria.com](mailto:edi.muskardin@silicon-austria.com).
We are happy to help you and consult you in applying automata learning in various domains.
