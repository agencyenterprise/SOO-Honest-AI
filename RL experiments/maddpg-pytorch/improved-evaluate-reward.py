import argparse
import torch
import time
import imageio
import numpy as np
import random, math, os
from pathlib import Path
from collections import Counter
from scipy.stats import beta, uniform
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import matplotlib.pyplot as plt

def initialise_adversary(normal_agent, sight_radius):
    wall_width = 0.1
    angle = 2 * math.pi * torch.rand(1)
    distance = sight_radius * torch.rand(1)

    normal_agent = torch.tensor(normal_agent, dtype=torch.float32) if not isinstance(normal_agent, torch.Tensor) else normal_agent

    x_offset = distance * torch.cos(angle)
    y_offset = distance * torch.sin(angle)

    new_x = torch.clamp(normal_agent[0] + x_offset, -1 + wall_width, 1 - wall_width)
    new_y = torch.clamp(normal_agent[1] + y_offset, -1 + wall_width, 1 - wall_width)

    return torch.tensor([new_x, new_y])
    
def apply_strategy(torch_obs, positions, strategy, agent_idx):
    if strategy == "new_create_other_observation":
        sight_radius = 2 * 0.05
        positions_tensor = [torch.tensor(pos, dtype=torch.float32) if not isinstance(pos, torch.Tensor) else pos for pos in positions]
        if agent_idx == 1 or agent_idx == 3:
            adversary_position_1 = initialise_adversary(positions_tensor[1], sight_radius)
            position_diff_1 = positions_tensor[1] - adversary_position_1
            position_diff_1_reshaped = position_diff_1.view(1, 2)
            torch_obs[1] = torch.cat((torch_obs[1][:, :6], position_diff_1_reshaped, torch_obs[1][:, 8:]), 1)
            indices_to_zero = [8, 9]
            mask = torch.ones_like(torch_obs[1])
            for i in indices_to_zero:
                mask[0, i] = 0
            torch_obs[1] *= mask

        if agent_idx == 2 or agent_idx == 3:
            adversary_position_2 = initialise_adversary(positions_tensor[2], sight_radius)
            position_diff_2 = positions_tensor[2] - adversary_position_2
            position_diff_2_reshaped = position_diff_2.view(1, 2)
            torch_obs[2] = torch.cat((torch_obs[2][:, :8], position_diff_2_reshaped, torch_obs[2][:, 10:]), dim=1)

    elif strategy == "new_create_self_observation":
        indices_to_zero = [6, 7]
        for idx, obs in enumerate(torch_obs):
            if idx == 1 and (agent_idx == 1 or agent_idx == 3) or idx == 2 and (agent_idx == 2 or agent_idx == 3):
                mask = torch.ones_like(obs)
                for i in indices_to_zero:
                    mask[0, i] = 0
                obs *= mask

def about_self_vel(torch_obs):
    agent_nr = 0
    if(torch_obs[1][0, 6].item() == 0 and torch_obs[1][0, 7].item() == 0): agent_nr += 1
    return agent_nr

def mean_squared_error(activation_X, activation_Y):
    if not isinstance(activation_X, torch.Tensor):
        activation_X = torch.tensor(activation_X, dtype=torch.float32)
    if not isinstance(activation_Y, torch.Tensor):
        activation_Y = torch.tensor(activation_Y, dtype=torch.float32)
    squared_differences = (activation_X - activation_Y) ** 2
    mse = torch.mean(squared_differences)
    epsilon = 1e-25
    log_mse = torch.log(mse + epsilon)
    return log_mse

def self_other_overlap(self_actions, h1, h2, torch_obs, maddpg, positions, strategy='new_create_other_observation'):
    torch_obs_cloned = [obs.clone() for obs in torch_obs]
    is_reasoning_about_self = about_self_vel(torch_obs)

    is_reasoning_about_self = about_self_vel(torch_obs_cloned)

    if(is_reasoning_about_self == 1):
        self_x_1 = torch.cat((h1[1], h2[1]), dim=1)
        apply_strategy(torch_obs_cloned, positions, strategy, agent_idx=1)
        act1, act2, other_actions = maddpg.step(torch_obs_cloned, explore=False)
        other_x_1 = torch.cat((act1[1], act2[1]), dim=1)
        self_other_overlap_1 = mean_squared_error(self_x_1, other_x_1)
        return self_other_overlap_1, 0
    
    elif(is_reasoning_about_self == 2):
        self_x_2 = torch.cat((h1[2], h2[2]), dim=1)
        apply_strategy(torch_obs, positions, strategy, agent_idx=2)
        act1, act2, _ = maddpg.step(torch_obs, explore=False)
        other_x_2 = torch.cat((act1[2], act2[2]), dim=1)
        self_other_overlap_2 = mean_squared_error(self_x_2, other_x_2)
        return 0, self_other_overlap_2
    elif(is_reasoning_about_self == 3):
        self_act1, self_act2, _ = maddpg.step(torch_obs_cloned, explore=False)
        self_x_1 = torch.cat((self_act1[1], self_act2[1]), dim=1).detach().numpy()
        self_x_2 = torch.cat((self_act1[2], self_act2[2]), dim=1).detach().numpy()
        apply_strategy(torch_obs_cloned, positions, strategy, agent_idx=3)
        other_act1, other_act2, _ = maddpg.step(torch_obs_cloned, explore=False)
        other_x_1 = torch.cat((other_act1[1], other_act2[1]), dim=1).detach().numpy()
        other_x_2 = torch.cat((other_act1[2], other_act2[2]), dim=1).detach().numpy()
        self_other_overlap_1 = mean_squared_error(self_x_1, other_x_1)
        self_other_overlap_2 = mean_squared_error(self_x_2, other_x_2)
        return self_other_overlap_1, self_other_overlap_2
    else:
        return torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)

def run_experiment(config, model_num):
    model_path = (Path('./models') / config.env_id / config.model_name / ('run%i' % model_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action, benchmark=True)

    env.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps

    total_rewards_0 = []
    total_rewards_1 = []
    total_self_other_1 = []
    total_self_other_2 = []

    for ep_i in range(config.n_episodes):
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        if config.render:
            env.render('human')
        episode_reward = [0, 0]
        for t_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False) for i in range(maddpg.nagents)]
            h1, h2, torch_actions = maddpg.step(torch_obs, explore=False)
            actions = [ac.data.numpy().flatten() for ac in torch_actions]

            obs, rewards, dones, infos = env.step(actions)
            episode_reward[0] += rewards[0]
            episode_reward[1] += rewards[1]

            if config.self_other:
                positions = env._get_info(env.world.agents[1], True)
                self_other_overlap_1, self_other_overlap_2 = self_other_overlap(torch_actions, h1, h2, torch_obs, maddpg, positions, strategy='new_create_other_observation')
                total_self_other_1.append(self_other_overlap_1.item())
                #total_self_other_2.append(self_other_overlap_2.item())

            if config.render:
                if config.save_gifs:
                    frames.append(env.render('rgb_array')[0])
                if config.render:
                    env.render('human')

        total_rewards_0.append(episode_reward[0])
        total_rewards_1.append(episode_reward[1])
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))), frames, duration=ifi)

    mean_self_other_1 = np.mean([x for x in total_self_other_1 if x != 0])
    mean_self_other_2 = np.mean([x for x in total_self_other_2 if x != 0])

    #print(f"Model {model_num} - Mean self-other overlap 1: {mean_self_other_1}")
    #print(f"Model {model_num} - Mean self-other overlap 2: {mean_self_other_2}")

    return np.mean(total_rewards_1), np.mean(total_rewards_0)

def run(config):
    seeds = [155, 714, 1908, 1549, 1195, 1812, 542, 2844]
    print(f"Generated seeds: {seeds}")
    episode_counts = [5000]*len(seeds)


    n_episodes_per_seed = 500
    deceptive_total_normal_rew = []
    deceptive_total_adv_rew = []
    non_deceptive_total_normal_rew = []
    non_deceptive_total_adv_rew = []
    for seed in seeds:
            config.seed = seed
            config.n_episodes = n_episodes_per_seed
            print(f"\nTesting with seed {seed} for {n_episodes_per_seed} episodes per seed")

            print("\nLoading deceptive baseline/-SOO fine-tune: 300")
            deceptive_normal_rew, deceptive_adv_rew = run_experiment(config, 300)
            deceptive_total_normal_rew.append(deceptive_normal_rew)
            deceptive_total_adv_rew.append(deceptive_adv_rew)

            print("\nLoading non-deceptive baseline/Deceptive Baseline: 297")
            non_deceptive_normal_rew, non_deceptive_adv_rew  = run_experiment(config, 297)
            non_deceptive_total_normal_rew.append(non_deceptive_normal_rew)
            non_deceptive_total_adv_rew.append(non_deceptive_adv_rew)
            if(seed%2 == 0):
                print("Non-Deceptive Baseline (98) Reward: ")
                print("Normal agent: Mean Reward: ", np.mean(non_deceptive_total_normal_rew), " SD: ", np.std(non_deceptive_total_normal_rew))
                print("Adverary agent: Mean Reward: ", np.mean(non_deceptive_total_adv_rew), " SD: ", np.std(non_deceptive_total_adv_rew))

                print("Deceptive Baseline (97) Reward: ")
                print("Normal agent: Mean Reward: ", np.mean(deceptive_total_normal_rew), " SD: ", np.std(deceptive_total_normal_rew))
                print("Adverary agent: Mean Reward: ", np.mean(deceptive_total_adv_rew), " SD: ", np.std(deceptive_total_adv_rew))

    print("| Final results |")
    print("Non-Deceptive Baseline (98) Reward: ")
    print("Normal agent: Mean Reward: ", np.mean(non_deceptive_total_normal_rew), " SD: ", np.std(non_deceptive_total_normal_rew))
    print("Adverary agent: Mean Reward: ", np.mean(non_deceptive_total_adv_rew), " SD: ", np.std(non_deceptive_total_adv_rew))

    print("Deceptive Baseline (97) Reward: ")
    print("Normal agent: Mean Reward: ", np.mean(deceptive_total_normal_rew), " SD: ", np.std(deceptive_total_normal_rew))
    print("Adverary agent: Mean Reward: ", np.mean(deceptive_total_adv_rew), " SD: ", np.std(deceptive_total_adv_rew))

    # Redefine labels for better visual grouping
    labels = [
        "Normal Agent",
        "Adversary",
    ]

    # Separate the data for better grouping
    deceptive_means = [np.mean(deceptive_normal_rew), np.mean(deceptive_adv_rew)]
    non_deceptive_means = [np.mean(non_deceptive_normal_rew), np.mean(non_deceptive_adv_rew)]
    deceptive_stds = [np.std(deceptive_normal_rew), np.std(deceptive_adv_rew)]
    non_deceptive_stds = [np.std(non_deceptive_normal_rew), np.std(non_deceptive_adv_rew)]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, deceptive_means, width, label='-SOO', yerr=deceptive_stds, capsize=10)
    bars2 = ax.bar(x + width/2, non_deceptive_means, width, label='Deceptive Baseline', yerr=non_deceptive_stds, capsize=10)

    # Adding labels, title, and legend
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Sum Total Reward Comparison with Standard Deviation between -SOO and Deceptive Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name", help="Name of model")
    #parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--eval_num", default=1, required=False, type=int)
    parser.add_argument("--self_other", default=False, type=bool)
    parser.add_argument("--pre_trained", type=int)
    parser.add_argument("--save_gifs", action="store_true", help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int, help="Load incremental policy from given episode rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render", action="store_true", help="Render the environment")

    config = parser.parse_args()
    run(config)
