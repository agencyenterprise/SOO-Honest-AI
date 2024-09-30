# RL Experiments for SOO-Honest-AI

This folder contains the scripts and files used for reinforcement learning (RL) experiments, as part of the paper *"Towards Safe and Honest AI Agents with Neural Self-Other Overlap."* These experiments aim to evaluate the impact of Self-Other Overlap (SOO) fine-tuning on RL agents' behaviors, particularly in reducing deceptive actions.

## Prerequisites

Before running the experiments, ensure you have the following dependencies installed:

### Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Gym
- Stable-Baselines3
- Multi-agent particle environments

You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Repository Structure

The RL experiments folder contains two main subdirectories:

### 1. **maddpg-pytorch**
   - This folder contains the PyTorch implementation of Multi-Agent Deep Deterministic Policy Gradients (MADDPG) with SOO fine-tuning integrated.
   - **Key Files**:
     - `fine-tune.py`: Script to fine-tune RL agents with SOO applied.
     - `evaluate.py`: Evaluates the performance of RL agents post-SOO fine-tuning.
     - `improved-evaluate-deception.py`: Evaluates deceptive behavior in agents.
     - `improved-evaluate-reward.py`: Assesses agents’ rewards after SOO fine-tuning.
     - `control_multiple_seed_plot_soo.py`: Control scripts for running multiple seeds and visualizing results.
     - Various plotting scripts such as `plot_sweep_metrics.py`, `plot_parameter_sweep.py` for visualizing experiment results.

### 2. **multiagent-particle-envs**
   - This folder contains the multi-agent particle environments used for training and evaluating the RL agents.
   - **Key Files**:
     - Various environment configuration files.
     - Assets used during training simulations.

## Running the Experiments

### 1. Fine-tuning the RL Model
Run the `fine-tune.py` script to fine-tune the RL agents with the Self-Other Overlap (SOO) fine-tuning technique.

```
python fine-tune.py
```

### 2. Evaluating the RL Model
Once the fine-tuning is complete, use the `evaluate.py` script to assess the agent's performance.

```
python evaluate.py
```

### 3. Evaluating Deception
To specifically evaluate the deceptive behavior in the agents post-SOO fine-tuning, use the `improved-evaluate-deception.py` script.

```
python improved-evaluate-deception.py
```

### 4. Visualizing Results
There are various plotting scripts available to visualize the results of different experiments:
- Use `plot_sweep_metrics.py` to visualize the parameter sweep results.
- Use `plot_parameter_sweep.py` to view the agents' performance over different parameters.

## Notes

- Ensure that the paths to the models and environments are set correctly in the scripts before running.
- GPU is highly recommended for fine-tuning and evaluation due to the computational intensity of the experiments.
- Run experiments with multiple seeds to ensure reproducibility.


For further information, please refer to the documentation in each script or contact us via email.
