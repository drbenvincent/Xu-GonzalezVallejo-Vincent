# Waiting in inter-temporal choice tasks affects discounting and subjective time perception

This repository contains data and analysis code in support of the paper

> Xu. P.,  Gonzalez-Vallejo, C. & Vincent, B. T. (under review) Waiting in inter-temporal choice tasks affects discounting and subjective time perception

We conduct Hierarchical Bayesian analysis on raw inter-temporal choice data using the methods outlined in Vincent (2016). The original implementation in Matlab is available at https://github.com/drbenvincent/delay-discounting-analysis, but here we use a Python based implementation using [PyMC3](https://docs.pymc.io) (Salvatier, Wiecki, & Fonnesbeck, 2016). We ran parameter recovery simulations, and parameter estimation from the data, for the modified Rachlin discount function (Vincent, & Stewart, 2019).

See the Jupyter notebooks (the `*.ipynb` files) to see the various analyses undertaken.

## References
Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55, https://doi.org/10.7717/peerj-cs.55

Vincent, B. T. (2016). Hierarchical Bayesian estimation and hypothesis testing for delay discounting tasks. Behavior Research Methods, 48(4), 1608–1620. http://doi.org/10.3758/s13428-015-0672-2

Vincent, B. T., & Stewart, N. (2019, January 30). The case of muddled units in temporal discounting. https://doi.org/10.31234/osf.io/29sgd

Xu. P.,  Gonzalez-Vallejo, C. & Vincent, B. T. (in preparation) The Waiting Experience Is Not What You Think
