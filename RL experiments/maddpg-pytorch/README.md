# RL Experiments

This folder contains the scripts and files used for reinforcement learning (RL) experiments, as part of the paper "**Towards Safe and Honest AI Agents with Neural Self-Other Overlap**." These experiments evaluate the impact of Self-Other Overlap (SOO) fine-tuning on RL agents' behaviors, particularly in reducing deceptive actions.

# MADDPG-PyTorch
PyTorch Implementation of MADDPG from [*Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275) (Lowe et. al. 2017)

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

### Dependencies

You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Repository Structure

This section outlines the main folders and files in the `maddpg-pytorch` folder. Please make sure to change your directory (`cd`) to `maddpg-pytorch` before running any commands below.

1. **maddpg-pytorch Folder**

   This folder contains the core codebase for the RL experiments.

2. **algorithms**

   Contains the implementation of the Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm with SOO fine-tuning.

3. **assets**

   Holds assets used for visualizations and plotting.

4. **models**

   Contains the trained models for RL agents, including both deceptive and honest baselines.

5. **utils**

   Utility scripts for managing environments and running experiments.

6. **train.py**

   The main script used to train RL agents (both deceptive and honest baselines).

7. **plot_quantitative_analysis.py**

   Script for generating quantitative analysis results after experiments.

8. **plot_parameter_sweep.py**

   Script for generating parameter sweep results, allowing you to assess performance across different experimental parameters.

9. **evaluate-deception.py**

   Script for evaluating deceptive behavior after the SOO fine-tuning.

10. **multiagent-particle-envs (RL Environment Folder)**

    This folder contains the multi-agent particle environments used for training and evaluating the RL agents in the experiments. These environments simulate decision-making scenarios where deceptive and honest behaviors are measured.


### Modified Physical Deception Environment

We conducted the RL experiment in a modified Physical Deception environment, featuring two agents and two landmarks: a goal landmark and a fake landmark. Both agents are rewarded for approaching the goal. The blue agent knows the landmarks' positions, while the "color-blind" red agent does not, leading it to follow the blue agent toward the goal. The red agent is trapped if it reaches the fake landmark. Agents know each other's positions only when they are within a predefined observation radius of each other. They are initialized randomly in the environment within each other's observation radius.

<img src="https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/hzt9gHpNwA2oHtwKX/r23gesvrxezvnedjqy7n" alt="Example GIF" width="500">

## Running the Experiments

Before running any of the following commands, ensure you are in the `maddpg-pytorch` folder:

```
cd maddpg-pytorch
```

### 1. Train Honest and Deceptive Baselines

To train the honest and deceptive baselines, use the following command:

```
python train.py simple_adversary MADDPG --n_episodes 40000 --episode_length 50
```

Make sure to check `simple_adversary.py` and ensure that the correct deceptive/non-deceptive reward is being used. Use `python train.py --help` for more information.

### 2. Perform SOO Fine-Tuning on the Deceptive Baseline

Once the deceptive baseline is trained, you can fine-tune it using SOO with the following command:

```
python train.py simple_adversary MADDPG --pre_trained <deceptive_baseline_run_num> --n_episodes 10000 --episode_length 50 --self_other True
```

Ensure the correct deceptive reward is being used in `simple_adversary.py` before running this command.

### 3. Generate Classification Results for Deceptive and Honest Baselines

To generate classification results for both deceptive and honest baselines, run the following command:

```
python improved-evaluate-deception.py simple_adversary MADDPG --run_deceptive <deceptive_baseline_run_num> --run_honest <honest_baseline_run_num> --episode_length 50
```

### 4. Visualizing Results

#### Parameter Sweep Results

Use `plot_parameter_sweep.py` to visualize parameter sweep results across multiple runs and seeds:

```
python plot_parameter_sweep.py --runs <deceptive_run_num> <SOO_run_num> <honest_run_num> --seeds 155 714 1908 1549 1195 1812 542 2844
```

#### Quantitative Analysis Results

To generate quantitative analysis results, use the following command:

```
python plot_quantitative_analysis.py --deceptive_run <deceptive_run_num> --honest_run <honest_run_num> --soo_run <SOO_run_num>
```

## Notes

- Ensure that the paths to the models and environments are set correctly in the scripts before running.
- GPU is highly recommended for fine-tuning and evaluation due to the computational intensity of the experiments.
- Run experiments with multiple seeds to ensure reproducibility.
