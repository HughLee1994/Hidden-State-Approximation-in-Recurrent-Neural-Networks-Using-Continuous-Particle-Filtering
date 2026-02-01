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

## Requirements

Core dependencies (see `requirements.txt` for the complete list):

```
torch>=1.10.0
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.7.0
```

---

## Running the Robot Localization Experiment

All training and evaluation parameters are specified via configuration files.

### Training

Run the following command to train the model:

```bash
python main.py -c ./configs/train.conf
```

The training configuration controls the model architecture, number of particles,
optimization settings, and dataset options.

---

### Evaluation

After training, save the latent particle representations:

```bash
python evaluate.py -c ./configs/eval.conf
```

This step generates particle tensors for downstream analysis.

---

### Particle Visualization

Visualize particle trajectories from the evaluation results:

```bash
python plot_particle.py --traj_num 0 --eval_num 0
```

* `traj_num`: trajectory index
* `eval_num`: evaluation run index

---

## Notes

* Please ensure the paths in the config files are correctly set.
* Evaluation must be run before particle visualization.
* Additional experiments can be configured by editing files under `configs/`.

```
```



## Results

| Task | Baseline LSTM | Our Method | Improvement |
|------|--------------|------------|-------------|
| Stock Prediction | - | - | - |
| Robot Localization | - | - | - |

*Fill in with your experimental results*

## Code Structure

```
.
├── models/
│   ├── particle_filter_rnn.py   # Main model implementation
│   ├── encoder_decoder.py        # Encoder-decoder variant
│   └── utils.py                  # Utility functions
├── experiments/
│   ├── stock_prediction.py
│   ├── robot_localization.py
│   └── sequence_benchmark.py
├── data/
│   └── preprocessing.py
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
└── requirements.txt
```

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work builds upon research in particle filtering, recurrent neural networks, and sequential Monte Carlo methods.
