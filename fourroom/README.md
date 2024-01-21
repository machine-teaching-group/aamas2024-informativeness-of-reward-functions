## [AAMAS 2024] Informativeness of Reward Functions in Reinforcement Learning

### Overview
This folder contains all the code files required for running numerical experiments for Room environment. The commands below generates data and plots in the folders ```results/``` and ```plots/```. 

### REINFORCE agent on Room environment
Run the following command to generate plot for Room environment using a single learner

```python teaching_fourroom.py  --use_pool=False --n_averaged=10```

Run the following command to generate plot for Room environment using a pool of learners

```python teaching_fourroom.py  --use_pool=True --n_averaged=10```
