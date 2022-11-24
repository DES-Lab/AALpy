<div align="center">
    <h1 align="center">AALpy</h1>
    <p align="center">An Active Automata Learning Library</p>

[![Python application](https://github.com/DES-Lab/AALpy/actions/workflows/python-app.yml/badge.svg)](https://github.com/DES-Lab/AALpy/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/DES-Lab/AALpy/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/DES-Lab/AALpy/actions/workflows/codeql-analysis.yml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aalpy)

[![GitHub issues](https://img.shields.io/github/issues/DES-Lab/AALpy)](https://github.com/DES-Lab/AALpy/issues)
![GitHub pull requests](https://img.shields.io/github/issues-pr/des-lab/aalpy)
[![Python 3.6](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/release/python-360/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/aalpy)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DES-Lab/AALpy/master)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
</div>
<hr />

AALpy is a light-weight active automata learning library written in pure Python. 
You can start learning automata in just a few lines of code. 

Whether you work with regular languages or you would like to learn models of 
(black-box) reactive systems, AALpy supports a wide range of modeling formalisms, including 
**deterministic**, **non-deterministic**, and **stochastic automata**. 

<div align="center">
	
| Automata Type   |      Supported Formalisms      |  Features |
|----------|:-------------:|------:|
| Deterministic     |  Deterministic Finite Automata<br />Mealy Machines<br />Moore Machines | Counterexample Processing<br />Seamless Caching<br />11 Equivalence Oracles <br /> Passive learning of all formalisms |
| Non-Deterministic |    Observable Non-Deterministic FSM <br /> Abstracted Non-Deterministic FSM|   Dynamic Observation Table Updates <br/> Size Reduction Trough Abstraction<br />|
| Stochastic        |  Markov Decision Processes<br />Stochastic Mealy Machines<br />Markov Chains |    Counterexample Processing<br />Row/Cell Compatability Metrics<br />Model Checking with PRISM<br />Alergia Passive Learning|

</div>

<!---
You can use it to learn **deterministic finite automata**, **Moore machines**, 
and **Mealy machines** of deterministic systems. 
If the system that you would like to learn shows non-deterministic or
stochastic behavior, AALpy allows you to learn **observable
nondeterministic finite-state machines**, **Markov decision processes**, 
or **stochastic Mealy machines**.

Finally, support for learning **abstracted non-deterministic Mealy machines** 
enables efficient learning of system models with large input space.
--->

AALpy enables efficient learning by providing a **large set of equivalence oracles**, implementing various conformance testing strategies. Learning 
is mostly based on Angluin's [L* algorithm](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf), for which AALpy supports a 
selection of optimizations, including **efficient counterexample processing** and **caching**.

AALpy also includes **passive automata learning algorithms**. Unlike active algorithms which learn by interaction with the system, passive
learning algorithms construct a model based on provided data.
AALpy includes an implementation of RPNI, a  passive deterministic automata learning algorithm that constructs a model that conforms to the
provided data.
AALpy also has an efficient implementation of the [ALERGIA](https://link.springer.com/article/10.1007/s10994-016-5565-9) algorithm, 
suited for passive learning of Markov Chains, Markov Decision processes, and Stochastic Mealy Machines.
For more efficient ALERGIA execution, AALpy offers bindings to [jAlergia](https://github.com/emuskardin/jAlergia), 
a Java twin of AALpy's ALERGIA code.
 
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the latest release of AALpy:
```bash
pip install aalpy

# or to install the current status of the master branch which might contain bugfixes/new features in between releases
pip install https://github.com/DES-Lab/AALpy/archive/master.zip
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
- <https://github.com/DES-Lab/AALpy/wiki>

For the **official documentation** of all classes and methods, check out:
- <https://des-lab.github.io/AALpy/documentation/index.html>

[Examples.py](https://github.com/DES-Lab/AALpy/blob/master/Examples.py) contains many examples and it is a great starting point. 

## Usage

All automata learning procedures follow this high-level approach:
- [Define the input alphabet and system under learning (SUL)](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems)
- [Choose the equivalence oracle](https://github.com/DES-Lab/AALpy/wiki/Equivalence-Oracles)
- [Run the learning algorithm](https://github.com/DES-Lab/AALpy/wiki/Setting-Up-Learning)

For more detailed examples, check out:
- [How to learn Regex with AALpy](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems#example---regexsul)
- [How to learn MQTT with AALpy](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems#example---mqtt)
- [Few Simple Examples](https://github.com/DES-Lab/Automata-Learning-Based-Diagnosis)
- [Interactive Examples](https://github.com/DES-Lab/AALpy/tree/master/notebooks)
- [Examples.py](https://github.com/DES-Lab/AALpy/blob/master/Examples.py)

[Examples.py](https://github.com/DES-Lab/AALpy/blob/master/Examples.py) contains examples covering almost the whole AALpy's functionality, and it is a great starting point/reference.
[Wiki](https://github.com/DES-Lab/AALpy/wiki) has a step-by-step guide to using AALpy and can help you understand AALpy and automata learning in general. 

<details>
  <summary>Code snipped demonstrating some of AALpy's functionalities</summary>

The following snippet demonstrates a short example in which an automaton is either [loaded](https://github.com/DES-Lab/AALpy/wiki/Loading,Saving,-Syntax-and-Visualization-of-Automata) or [randomly generated](https://github.com/DES-Lab/AALpy/wiki/Generation-of-Random-Automata) and then [learned](https://github.com/DES-Lab/AALpy/wiki/Setting-Up-Learning).
```python
from aalpy.utils import load_automaton_from_file, save_automaton_to_file, visualize_automaton, generate_random_dfa, dfa_from_state_setup
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_Lstar

# load an automaton
# automaton = load_automaton_from_file('path_to_the_file.dot', automaton_type='dfa')

# or construct it from state setup
dfa_state_setup = {
    'q0': (True, {'a': 'q1', 'b': 'q2'}),
    'q1': (False, {'a': 'q0', 'b': 'q3'}),
    'q2': (False, {'a': 'q3', 'b': 'q0'}),
    'q3': (False, {'a': 'q2', 'b': 'q1'})
}

small_dfa = dfa_from_state_setup(dfa_state_setup)

# or randomly generate one
random_dfa = generate_random_dfa(alphabet=[1,2,3,4,5],num_states=20, num_accepting_states=8)
big_random_dfa = generate_random_dfa(alphabet=[1,2,3,4,5],num_states=2000, num_accepting_states=500)

# get input alphabet of the automaton
alphabet = random_dfa.get_input_alphabet()

# loaded or randomly generated automata are considered as BLACK-BOX that is queried
# learning algorithm has no knowledge about its structure
# create a SUL instance for the automaton/system under learning
sul = DfaSUL(random_dfa)

# define the equivalence oracle
eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09)

# start learning
learned_dfa = run_Lstar(alphabet, sul, eq_oracle, automaton_type='dfa')

# save automaton to file and visualize it
# save_automaton_to_file(learned_dfa, path='Learned_Automaton', file_type='dot')
# or
learned_dfa.save()

# visualize automaton
# visualize_automaton(learned_dfa)
learned_dfa.visualize()
# or just print its DOT representation
print(learned_dfa)
```

</details>

To make experiments reproducible, define a random seed at the beginning of your program.
```Python
from random import seed
seed(2) # all experiments will be reproducible
```

## Selected Applications
AALpy has been used to:
- [Learn Bluetooth Low-Energy](https://github.com/apferscher/ble-learning)
- [Learn Input-Output Behavior of RNNs](https://github.com/DES-Lab/Extracting-FSM-From-RNNs)
- [Find bugs in VIM text editor](https://github.com/DES-Lab/AALpy/discussions/13)

## Cite AALpy and Research Contact
If you use AALpy in your research, please cite us with of the following:
- [Extended version (preferred)](https://www.researchgate.net/publication/359517046_AALpy_an_active_automata_learning_library/citation/download)
- [Tool paper](https://dblp.org/rec/conf/atva/MuskardinAPPT21.html?view=bibtex)

If you have research suggestions or you need specific help concerning your research, feel free to start a [discussion](https://github.com/DES-Lab/AALpy/discussions) or contact [edi.muskardin@silicon-austria.com](mailto:edi.muskardin@silicon-austria.com).
We are happy to help you and consult you in applying automata learning in various domains.

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.
In case of any questions or possible bugs, please open issues.
