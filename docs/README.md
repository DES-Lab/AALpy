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

For more detailed examples, check out:
- [How to learn Regex with AALpy](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface%2C-or-How-to-Learn-Your-Systems/_edit#example---regexsul)
- [How to learn MQTT with AALpy](https://github.com/DES-Lab/AALpy/wiki/SUL-Interface,-or-How-to-Learn-Your-Systems#example---mqtt)
- [Interactive Examples](https://github.com/DES-Lab/AALpy/tree/master/notebooks)
- [Examples.py](https://github.com/DES-Lab/AALpy/blob/master/Examples.py)

With AALpy, you can
 - [Load, save and visualize automata](https://github.com/DES-Lab/AALpy/wiki/Loading,Saving,-Syntax-and-Visualization-of-Automata)
 - [Randomly generated automata](https://github.com/DES-Lab/AALpy/wiki/Generation-of-Random-Automata)


## Research Contact
If you have research suggestions or need specific help concerning your research, feel free to contact [edi.muskardin@silicon-austria.com](mailto:edi.muskardin@silicon-austria.com).
We are happy to help you and consult you in applying automata learning in various domains.
