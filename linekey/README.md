## [AAMAS 2024] Informativeness of Reward Functions in Reinforcement Learning

### Overview
This folder contains all the code files required for running numerical experiments for LineK environment. The commands below generates data and plots in the folders ```results/``` and ```plots/```. 

### REINFORCE agent on LineK environment
Run the following command to generate plot for LineK environment using a single learner 

```python teaching_linekey.py --use_pool=False --n_averaged=10 ```

Run the following command to generate plot for LineK environment using a pool of learners 

```python teaching_linekey.py --use_pool=True --n_averaged=10 ```