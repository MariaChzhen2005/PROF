# Name

This is the official repository that implements the following paper:

...

[[slides]](docs/slides.pdf)[[paper]](https://dl.acm.org/doi/10.1145/3408308.3427980)[[video]](https://www.youtube.com/watch?v=rH64WyPHCVE) 

# Overview

**Framework.** 

# Code Usage
### Clone repository
```
git clone https://github.com/INFERLab/PROF.git
cd COHORT
```

### Set up the environment 
Set up the virtual environment with your preferred environment/package manager.

The instruction here is based on **conda**. ([Install conda](https://docs.anaconda.com/anaconda/install/))
```
conda create --name cohort-env python=3.7 -c conda-forge -f requirements.txt
condo activate cohort-env
```

### File Structure
```
.
├── agents
│   ├── base.py             # Implement a controller that creates the CVXPY problem given building parameters
│   └── nn_policy.py        # Inherit the controller from base.py; Forward pass: NN + Differentiable projection
├── algo                    
│   └── ppo.py	 	    # A PPO trainer 
├── env
│   └── inverter.py         # Implements the IEEE 37-bus case
├── utils
│   ├── network.py          # Implements vanilla MLP and LSTM
│   └── ppo_utils.py        # Helper function for PPO trainer, e.g. Replay_Memory, Advantage_func
├── network		    # Matlab code for linearization; Data for grid;
└── mypypower		    # Include some small changes from PyPower source code

```

### Running
You can replicate our experiments for *Energy efficiency in a single building* with `main_IW.py`.


### Feedback

Feel free to send any questions/feedback to: [Bingqing Chen](mailto:bingqinc@andrew.cmu.edu)

### Citation

If you use COHORT, please cite us as follows:

```

```
