AALpy is a light-weight automata learning library written in Python. 
You can start learning models of black-box systems with a few lines of code.

AALpy supports both **active** and **passive** automata learning algorithms that can be used to learn a variety of modeling formalisms, including 
**deterministic**, **non-deterministic**, and **stochastic automata**, as well as **deterministic context-free grammars/pushdown automata**.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the latest release of AALpy:
```bash
pip install aalpy
```
To install current version of the master branch (it might contain bugfixes and added functionalities between releases):
```bash
pip install https://github.com/DES-Lab/AALpy/archive/master.zip
```
The minimum required version of Python is 3.6.  
Ensure that you have [Graphviz](https://graphviz.org/) installed and added to your path if you want to visualize models.

For manual installation, clone the repo and install `pydot` (the only dependency).

## Documentation and Wiki

If you are interested in automata learning or would like to understand the automata learning process in more detail,
please check out our **Wiki**. On Wiki, you will find more detailed examples on how to use AALpy.
- <https://github.com/DES-Lab/AALpy/wiki>

***[Examples.py](https://github.com/DES-Lab/AALpy/blob/master/Examples.py)*** contains examples covering almost the whole of AALpy's functionality and its a great starting point. 

### Usage

All active automata learning procedures follow this high-level approach:
- [Define the input alphabet and system under learning (SUL)](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems)
- [Choose the equivalence oracle](https://github.com/DES-Lab/AALpy/wiki/Equivalence-Oracles)
- [Run the learning algorithm](https://github.com/DES-Lab/AALpy/wiki/Setting-Up-Learning)

Passive learning algorithm simply require you to provide data in the appropriate format (check Wiki and Examples) and run the learning function.

## Selected Applications
AALpy has been used to:
- [Learn Models of Bluetooth Low-Energy](https://github.com/apferscher/ble-learning)
- [Find bugs in VIM text editor](https://github.com/DES-Lab/AALpy/discussions/13)
- [Learn Input-Output Behavior of RNNs](https://github.com/DES-Lab/Extracting-FSM-From-RNNs)
- [Learn Models of GIT](https://github.com/taburg/git-learning)
- [Solve RL Problems](https://github.com/DES-Lab/Learning-Environment-Models-with-Continuous-Stochastic-Dynamics)

## Cite AALpy and Research Contact
If you use AALpy in your research, please cite us with of the following:
- [Extended version (preferred)](https://www.researchgate.net/publication/359517046_AALpy_an_active_automata_learning_library/citation/download)
- [Tool paper](https://dblp.org/rec/conf/atva/MuskardinAPPT21.html?view=bibtex)

If you have research suggestions or you need specific help concerning your research, feel free to start a [discussion](https://github.com/DES-Lab/AALpy/discussions) or contact [edi.muskardin@silicon-austria.com](mailto:edi.muskardin@silicon-austria.com).
We are happy to help you and consult you in applying automata learning in various domains.

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.
In case of any questions or possible bugs, please open issues.
