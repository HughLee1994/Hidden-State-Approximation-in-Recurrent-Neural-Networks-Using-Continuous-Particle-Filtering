# Hidden-State-Approximation-in-Recurrent-Neural-Networks-Using-Continuous-Particle-Filtering
implementation of the paper "Hidden State Approximation in Recurrent Neural Networks Using Continuous Particle Filtering" 

**Paper:** [arxiv.org/abs/2212.09008](https://arxiv.org/abs/2212.09008)

## Overview

This repository provides the implementation of a novel approach to recurrent neural networks (RNNs) that uses particle filtering to approximate the distribution of hidden states, rather than maintaining them deterministically as in traditional RNNs.



````markdown
## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/HughLee1994/Hidden-State-Approximation-in-Recurrent-Neural-Networks-Using-Continuous-Particle-Filtering.git
cd particle-filtering-rnn

# (Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---



### Training

Run the following command to train the model:

```bash
python main.py 
```

---


## Citation

If you find this work useful, please cite:

```bibtex
@article{li2022hidden,
  title={Hidden State Approximation in Recurrent Neural Networks Using Continuous Particle Filtering},
  author={Li, Dexun},
  journal={arXiv preprint arXiv:2212.09008},
  year={2022}
}
```

