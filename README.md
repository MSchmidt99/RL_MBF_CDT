# RL_MBF_CDT
This repository utilizes the architecture posed by the paper ["CDT: Cascading Decision Trees for Explainable Reinforcement Learning" by Zihan Ding et al](https://arxiv.org/abs/2011.07553), and encapsulates it within a boosted forest architecture for a light-weight methodology that can create powerful models for accurate feature importances.

This architecture is applied to a stock trading environment which utilizes an easily swappable strategy to distill the state space down to strong indicators for positive, negative, or neutral price movements.

TODO:
 - refactor CDT.py to allow for cascading multiple feature learning trees onto each-other
