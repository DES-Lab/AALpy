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

<center>

| Automata Type   |      Supported Formalisms      |  Features |
|----------|:-------------:|------:|
| Deterministic     |  Deterministic Finite Automata<br />Mealy Machines<br />Moore Machines | Counterexample Processing<br />Seamless Caching<br />11 Eq. Oracles |
| Non-Deterministic |    Observable Non-Deterministic FSM <br /> Abstracted Non-Deterministic FSM|   Size Reduction Trough Abstraction<br />|
| Stochastic        |  Markov Decision Processes<br />Stochastic Mealy Machines |    Counterexample Processing<br />Row/Cell Compatability Metrics<br />Model Checking with PRISM|

</center>

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
- <https://github.com/DES-Lab/AALpy/wiki>

For the **official documentation** of all classes and methods, check out:
- <https://des-lab.github.io/AALpy/documentation/index.html>

**Interactive examples** can be found in the [notebooks](https://github.com/DES-Lab/AALpy/tree/master/notebooks) folder.
If you would like to interact/change those examples in the browser, click on the following badge. (Navigate to the _notebooks_ folder and select one notebook)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DES-Lab/AALpy/master)

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

The following snippet demonstrates a short example in which an automaton is either [loaded](https://github.com/DES-Lab/AALpy/wiki/Loading,Saving,-Syntax-and-Visualization-of-Automata) or [randomly generated](https://github.com/DES-Lab/AALpy/wiki/Generation-of-Random-Automata) and then [learned](https://github.com/DES-Lab/AALpy/wiki/Setting-Up-Learning).
```python
from aalpy.utils import load_automaton_from_file, save_automaton_to_file, visualize_automaton, generate_random_dfa
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_Lstar

# load an automaton
# automaton = load_automaton_from_file('path_to_the_file.dot', automaton_type='dfa')

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
save_automaton_to_file(learned_dfa, path='Learned_Automaton', file_type='dot')

# visualize automaton
visualize_automaton(learned_dfa)
# or just print its DOT representation
print(learned_dfa)
```

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
If you use AALpy in your research, please cite:
```
@inproceedings{aalpy,
	title = {{AALpy}: An Active Automata Learning Library},
	author = {Edi Mu\v{s}kardin and Bernhard K. Aichernig and Ingo Pill and Andrea Pferscher and Martin Tappler},
	booktitle = {Automated Technology for Verification and Analysis - 19th International
	Symposium, {ATVA} 2021, Gold Coast, Australia, October 18-22, 2021, Proceedings},
	series    = {Lecture Notes in Computer Science},  
	publisher = {Springer},
	year      = {2021},
}
```
If you have research suggestions or you need specific help concerning your research, feel free to start a [discussion](https://github.com/DES-Lab/AALpy/discussions) or contact [edi.muskardin@silicon-austria.com](mailto:edi.muskardin@silicon-austria.com).
We are happy to help you and consult you in applying automata learning in various domains.

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.
In case of any questions or possible bugs, please open issues.
