import argparse
import torch
import time
import imageio
import numpy as np
import random, math, os
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy.stats import beta, uniform
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from scipy.stats import truncnorm

def initialise_adversary(normal_agent, sight_radius):
    wall_width = 0.1
    angle = 2 * math.pi * torch.rand(1)  # Uniform random angle
    distance = sight_radius * torch.rand(1)  # Uniform random distance within sight radius

    # Ensure normal_agent is a tensor
    normal_agent = torch.tensor(normal_agent, dtype=torch.float32) if not isinstance(normal_agent, torch.Tensor) else normal_agent

    x_offset = distance * torch.cos(angle)
    y_offset = distance * torch.sin(angle)

    # Calculate new position and ensure it's within bounds
    new_x = torch.clamp(normal_agent[0] + x_offset, -1 + wall_width, 1 - wall_width)
    new_y = torch.clamp(normal_agent[1] + y_offset, -1 + wall_width, 1 - wall_width)

    # Return new position as a tensor
    return torch.tensor([new_x, new_y])
    
    
def apply_strategy(torch_obs, positions, strategy, agent_idx):
    
    if strategy=="new_create_other_observation":
        #torch.autograd.set_detect_anomaly(True)
        sight_radius = 2*0.05
        # Ensure positions are converted to PyTorch tensors
        positions_tensor = [torch.tensor(pos, dtype=torch.float32) if not isinstance(pos, torch.Tensor) else pos for pos in positions]
        #print(positions_tensor)
        if agent_idx == 1 or agent_idx == 3:
            # Assuming initialise_adversary returns a PyTorch tensor for adversary_position
            adversary_position_1 = initialise_adversary(positions_tensor[1], sight_radius)
            #print("adversary_position_1: ", adversary_position_1)
            # Perform subtraction using PyTorch tensors
            #print("positions_tensor: ", positions_tensor)
            #print("positions_tensor[1]: ", positions_tensor[1])
            position_diff_1 = positions_tensor[1] - adversary_position_1
            position_diff_1_reshaped = position_diff_1.view(1, 2)
            # Clone torch_obs[1] to avoid in-place operations issues and update it
            #torch_obs[1] = torch_obs[1].clone()
            #print("torch_obs[1]: ", torch_obs[1])
            #print("torch_obs[1].shape: ", torch_obs[1].shape)
            #torch_obs[1][0, 6:8] = position_diff_1_reshaped
            ##torch_obs[1] = torch.cat((torch_obs[1][:, :6], position_diff_1_reshaped, torch_obs[1][:, 8:]), 1)
            torch_obs[1] = torch.cat((torch_obs[1][:, :6], position_diff_1_reshaped, torch_obs[1][:, 8:]), 1)
            indices_to_zero = [8,9]
            #print("torch obs before: ", torch_obs[1])
            mask = torch.ones_like(torch_obs[1])
            for i in indices_to_zero:
                mask[0, i] = 0
                # Apply the mask
            torch_obs[1] *= mask
            #print("torch obs after: ", torch_obs[1])
            

        if agent_idx == 2 or agent_idx == 3:
            adversary_position_2 = initialise_adversary(positions_tensor[2], sight_radius)
            position_diff_2 = positions_tensor[2] - adversary_position_2
            position_diff_2_reshaped = position_diff_2.view(1, 2)  
            #torch_obs[2] = torch_obs[2].clone()
            #torch_obs[2][0, 6:8] = position_diff_2_reshaped
            ##torch_obs[2] = torch.cat((torch_obs[2][:, :6], position_diff_2_reshaped, torch_obs[2][:, 8:]), dim=1)
            torch_obs[2] = torch.cat((torch_obs[2][:, :8], position_diff_2_reshaped, torch_obs[2][:, 10:]), dim=1)
            #indices_to_zero = [17,16]
            #mask = torch.ones_like(torch_obs[2])
            #for i in indices_to_zero:
            #    mask[0, i] = 0
            #    # Apply the mask
            #torch_obs[2] *= mask
            
    elif strategy=="new_create_self_observation":
        indices_to_zero = [6, 7]
        #indices_to_zero = [8, 9, 12, 13, 10, 11, 14, 15]
        for idx, obs in enumerate(torch_obs):
            if idx == 1 and (agent_idx == 1 or agent_idx == 3) or idx == 2 and (agent_idx == 2 or agent_idx == 3):
                # Create a mask that will be used to zero out selected indices
                mask = torch.ones_like(obs)
                for i in indices_to_zero:
                    mask[0, i] = 0
                # Apply the mask
                obs *= mask
        #print("before torch obs: ", torch_obs)
        #torch_obs[1] = torch.cat((torch_obs[1][:, :8], torch.tensor(np.array([1,1])).float().view(1, 2), torch_obs[1][:, 10:]), 1)
        #print("after torch obs: ", torch_obs)
            

def about_self_vel(torch_obs):
    #here by self we mean normal agent 
    agent_nr = 0
    if(torch_obs[1][0, 6].item() == 0 and torch_obs[1][0, 7].item() == 0): agent_nr += 1
    #if((torch_obs[2][0, 8].item() == 0 and torch_obs[2][0, 9].item() == 0) and (torch_obs[2][0, 12].item() == 0 and torch_obs[2][0,13].item() == 0)): agent_nr += 2
    return agent_nr


def mean_squared_error(activation_X, activation_Y):
    # Convert inputs to tensors if they are not already, ensuring they are of the same type
    if not isinstance(activation_X, torch.Tensor):
        activation_X = torch.tensor(activation_X, dtype=torch.float32)
    if not isinstance(activation_Y, torch.Tensor):
        activation_Y = torch.tensor(activation_Y, dtype=torch.float32)
    
    # Compute the squared differences
    squared_differences = (activation_X - activation_Y) ** 2
    
    # Calculate the mean of these squared differences
    mse = torch.mean(squared_differences)
    
    # Adding a small epsilon to the mse before taking log to prevent log(0)
    epsilon = 1e-25
    log_mse = torch.log(mse + epsilon)
    
    return log_mse

def self_other_overlap(self_actions, h1, h2, torch_obs, maddpg, positions, strategy='new_create_other_observation'):    
    torch_obs_cloned = [obs.clone() for obs in torch_obs]
    is_reasoning_about_self = about_self_vel(torch_obs)
    #print("Before: ", is_reasoning_about_self)
    # Make the obseervation of both agents into a self observation 
    #if(is_reasoning_about_self==0):
    #    torch_obs_cloned = [obs.clone() for obs in torch_obs]
    #    apply_strategy(torch_obs_cloned, positions, 'new_create_self_observation', agent_idx=3)

    #if(is_reasoning_about_self==1):
    #    torch_obs_cloned = [obs.clone() for obs in torch_obs]
    #    apply_strategy(torch_obs_cloned, positions, 'new_create_self_observation', agent_idx=2)

    #if(is_reasoning_about_self==2):
    #    torch_obs_cloned = [obs.clone() for obs in torch_obs]
    #    apply_strategy(torch_obs_cloned, positions, 'new_create_self_observation', agent_idx=1)

        
    is_reasoning_about_self = about_self_vel(torch_obs_cloned)  
    #print(is_reasoning_about_self)
    #print("After: ", is_reasoning_about_self)
    #if(is_reasoning_about_self!=1): 
    #    print("Transforming the observation to a self observation failed!")
    #    print("torch_obs: ", torch_obs)

         
    if(is_reasoning_about_self==1):
        #self_act1, self_act2, self_actions = maddpg.step(torch_obs_cloned, explore=False)
        #self_x_1 = torch.cat((self_act1[1], self_act2[1]), dim=1).detach().numpy()
        self_x_1 = torch.cat((h1[1], h2[1]), dim=1)
        
        apply_strategy(torch_obs_cloned, positions, strategy, agent_idx=1)
        
        act1, act2, other_actions = maddpg.step(torch_obs_cloned, explore=False)
        other_x_1 = torch.cat((act1[1], act2[1]), dim=1)
        #print("self_Actions: ", self_actions[1])
        #print("other_actions: ", other_actio`ns[1])
        #self_other_overlap_1 = mean_squared_error(self_x_1, other_x_1)
        self_other_overlap_1 = mean_squared_error(self_actions, other_actions[1])
        #self_other_overlap_1 = pca_overlap(self_x_1, other_x_1)
        return self_other_overlap_1, 0
    elif(is_reasoning_about_self==2):
        self_x_2 = torch.cat((h1[2], h2[2]), dim=1)
        apply_strategy(torch_obs, positions, strategy, agent_idx=2)
        act1, act2, _ = maddpg.step(torch_obs, explore=False)
        other_x_2 = torch.cat((act1[2], act2[2]), dim=1)
        self_other_overlap_2 = mean_squared_error(self_x_2, other_x_2)
        return 0, self_other_overlap_2
    elif(is_reasoning_about_self==3):
        #torch.autograd.set_detect_anomaly(True)
        #For last hidden layer only (self)
        #self_x_1 = h2[1]
        #self_x_2 = h2[2]
        #For both hidden layers
        self_act1, self_act2, _ = maddpg.step(torch_obs_cloned, explore=False)
        #Uset .detach().numpy() for random runs
        self_x_1 = torch.cat((self_act1[1], self_act2[1]), dim=1).detach().numpy()
        self_x_2 = torch.cat((self_act1[2], self_act2[2]), dim=1).detach().numpy()

        apply_strategy(torch_obs_cloned, positions, strategy, agent_idx=3)

        other_act1, other_act2, _ = maddpg.step(torch_obs_cloned, explore=False)
        #For last hidden layer only (other)
        #other_x_1 = act2[1]
        #other_x_2 = act2[2]
        #For both hidden layers
        other_x_1 = torch.cat((other_act1[1], other_act2[1]), dim=1).detach().numpy()
        other_x_2 = torch.cat((other_act1[2], other_act2[2]), dim=1).detach().numpy()
        #print("other_x_1: ", other_x_1)
        
        self_other_overlap_1 = mean_squared_error(self_x_1, other_x_1)
        self_other_overlap_2 = mean_squared_error(self_x_2, other_x_2)
        #print("self_other_overlap_1: ",self_other_overlap_1)
        return self_other_overlap_1, self_other_overlap_2
    else:
        return torch.tensor(0, dtype=torch.float32),torch.tensor(0, dtype=torch.float32)


def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action, benchmark = True)
    pretrained_maddpg = MADDPG.init_from_save(Path('./models') / config.env_id / config.model_name / ('run%i' % config.pre_trained) / 'model.pt') if config.pre_trained else None
    # Set seed
    env.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    total_rewards_0 = []
    total_rewards_1 = []
    total_rewards_2 = []
    
    list_mean_1 = []
    list_mean_2 = []
    
    total_self_other_1 = []
    total_self_other_2 = []
    
    mean_adversary_benchmarks, mean_good_agent_benchmarks = [], [[], []]

    count_to_goal = 0
    count_to_nongoal = 0
    count_deception = 0
    
    normal_at_goal = 0  
    adv_at_nongoal = 0
    adv_at_goal = 0
    
    total_non_goal_angle_list = []
    
    total_threshold_list = []
    
    total_action_list = []
    
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        #env.set_year(config.year)
        episode_reward = [0,0,0]
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        if(config.render):
            env.render('human')
            
        ep_mean_self_other_1, ep_mean_self_other_2 = [], []
        ep_benchmark_adversary, ep_benchmark_good_agents = 0, [0] * 2
        adversary_benchmarks, good_agent_benchmarks = [], [[], []]
        distance_to_goal, distance_to_landmark = [0,0], [0,0]
        
        if config.pre_trained:
            pretrained_maddpg.prep_rollouts(device='cpu')
        
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            #if(config.self_other_strategy):
            #    other_in_sight = about_other_vel(torch_obs)
            #    if(other_in_sight==1): 
            #        apply_strategy(torch_obs, None, 'create_self_observation', agent_idx=1)
            #    if(other_in_sight==2): 
            #        apply_strategy(torch_obs, None, 'create_self_observation', agent_idx=2)
            #    if(other_in_sight==3): 
            #        apply_strategy(torch_obs, None, 'create_self_observation', agent_idx=3)
            # get actions as torch Variables
            h1, h2, torch_actions = maddpg.step(torch_obs, explore=False)
            #print("torch_actions: ", torch_actions)
            #print(h1)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            #print("actions: ", actions)
            
            if config.pre_trained:
                pretrained_h1, pretrained_h2, pretrained_torch_actions = pretrained_maddpg.step(torch_obs, explore=False)
                #kl_div_0 = calculate_kl_divergence(pretrained_torch_actions[0], torch_actions[0])
                #print("pretrained agent actions:", pretrained_torch_agent_actions)
                #kl_div_1 = calculate_kl_divergence(pretrained_torch_actions[1], torch_actions[1])
                #print("pretrained_agent_actions[1] :", pretrained_torch_agent_actions[1])
                #print("kl_div_1", kl_div_1)
                #kl_div_2 = calculate_kl_divergence(pretrained_torch_actions[2], torch_actions[2])
                #print("kl_div_2", kl_div_2)
                #print("kl div between the same tensors: ", calculate_kl_divergence(pretrained_torch_agent_actions[2],pretrained_torch_agent_actions[2]))

                pretrained_agent_actions = [ac.data.numpy() for ac in pretrained_torch_actions]
                #print("agent_actions:", agent_actions)
            for agent_idx, agent in enumerate(env.world.agents):
                benchmark_data = env._get_info(agent)
                if agent.adversary:
                    if(config.eval_num):
                        if(t_i>=config.episode_length/2):
                            ep_benchmark_adversary += benchmark_data
                    else:
                        ep_benchmark_adversary += benchmark_data
                else:
                    distance_to_goal[agent_idx-1] = next((num for num, count in Counter(benchmark_data).items() if count > 1), 0) 
                    #repeats twice as it is both a landmark and a goal
                    distance_to_landmark[agent_idx-1] = next((num for num, count in Counter(benchmark_data).items() if count == 1), 0) #distance to non-goal landmark
                                    
                    # Find index of the agent closest to the goal
                    closest_to_goal_idx = 0 if distance_to_goal[0] < distance_to_goal[1] else 1
                    # Find index of the other agent (the one not closest to the goal)
                    other_agent_idx = 1 - closest_to_goal_idx
                    if(config.eval_num):
                        if(t_i>=config.episode_length/2):
                            ep_benchmark_good_agents[0] += distance_to_goal[closest_to_goal_idx]
                            ep_benchmark_good_agents[1] += distance_to_landmark[other_agent_idx]
                    else:
                        ep_benchmark_good_agents[0] += distance_to_goal[closest_to_goal_idx]
                        ep_benchmark_good_agents[1] += distance_to_landmark[other_agent_idx]
            
            #print(actions)
            obs, rewards, dones, infos = env.step(actions)
            episode_reward[0] += rewards[0]
            episode_reward[1] += rewards[1]
            if(t_i==config.episode_length-1):
                positions  = env._get_info(env.world.agents[1], True)
                #print(positions)
                #count_to_goal += positions[2]
                #if(positions[3]>0):
                count_to_nongoal += positions[2]
                if(positions[2]>0): print("DECEPTIVE EPISODE: ", ep_i)
                adv_at_goal += positions[3]
                adv_at_nongoal += positions[4]
                normal_at_goal += positions[5]
                total_non_goal_angle_list.append(positions[6])
                total_threshold_list.append(positions[7])
                total_action_list.append(positions[8])
            #episode_reward[2] += rewards[2]
            if(config.self_other):
                positions  = env._get_info(env.world.agents[1], True)
                #print(positions)
                #print("AAA")
                #self_other_overlap_1, self_other_overlap_2 = self_other_overlap(h1, h2, torch_obs, maddpg, positions, strategy='other_short_sighted_adversary')
                ep_mean_self_other_1.append(self_other_overlap_1)
                ep_mean_self_other_2.append(self_other_overlap_2)
                #rewards[0] -= kl_div_0
                #rewards[1] = rewards[1] - self_other_overlap_1 - 2*kl_div_1 #previously +=
                #rewards[2] = rewards[2] - self_other_overlap_2 - 2*kl_div_2 
                #rewards[1] = rewards[1] + 2*normalise(self_other_overlap_1, "inverted_exponential") #previously +=
                #rewards[2] = rewards[2] + 2*normalise(self_other_overlap_2, "inverted_exponential")  #previously +=
                list_mean_1.append(self_other_overlap_1)
                list_mean_2.append(self_other_overlap_2)
            if(config.random):
                positions  = env._get_info(env.world.agents[1], True)
                self_other_overlap_1, self_other_overlap_2 = self_other_overlap(h1, h2, torch_obs, maddpg, positions, strategy='other_short_sighted_adversary')
                ep_mean_self_other_1.append(self_other_overlap_1)
                ep_mean_self_other_2.append(self_other_overlap_2)
                #control_value_1 = adjusted_control(self_other_overlap_1, np.mean(ep_mean_self_other_1),  np.std(ep_mean_self_other_1))
                #control_value_2 = adjusted_control(self_other_overlap_2, np.mean(ep_mean_self_other_2), np.std(ep_mean_self_other_2))
                #rewards[1] -= control_value_1 #previously +=
                #rewards[2] -= control_value_2 #previously +=
                total_self_other_1.append(self_other_overlap_1)
                total_self_other_2.append(self_other_overlap_2)
                #control_value_1 = custom_adjusted_control(self_other_overlap_1, mean=np.mean(total_self_other_1),  std=np.std(total_self_other_1), upper_bound=max(total_self_other_1))
                #control_value_2 = custom_adjusted_control(self_other_overlap_2, mean=np.mean(total_self_other_2), std=np.std(total_self_other_2), upper_bound=max(total_self_other_2))
                #rewards[1] = rewards[1] + 2*normalise(control_value_1, "inverted_exponential")
                #rewards[2] = rewards[2] + 2*normalise(control_value_2, "inverted_exponential")
                list_mean_1.append(self_other_overlap_1)
                list_mean_2.append(self_other_overlap_2)
                

            if(config.render):
                if config.save_gifs:
                    frames.append(env.render('rgb_array')[0])
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
            if(config.render): env.render('human')
        total_rewards_0.append(episode_reward[0])
        total_rewards_1.append(episode_reward[1])
        #total_rewards_2.append(episode_reward[2])
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
            
        adversary_benchmarks.append(ep_benchmark_adversary)
        mean_adversary_benchmarks.append(np.mean(adversary_benchmarks))
        for i in range(len(ep_benchmark_good_agents)):
            good_agent_benchmarks[i].append(ep_benchmark_good_agents[i])
            mean_good_agent_benchmarks[i].append(np.mean(good_agent_benchmarks[i]))
            
    mean_rewards = [np.mean(total_rewards_0),np.mean(total_rewards_1)]
    

    print(f"Mean Reward over {config.n_episodes} episodes of agent 0 (Adversary): {mean_rewards[0]} with std: {np.std(total_rewards_0)}")
    print(f"Mean Reward over {config.n_episodes} episodes of agent 1 (Good): {mean_rewards[1]} with std: {np.std(total_rewards_1)}")
    print(f"Acted deceptively : {(count_to_nongoal)}")
    print(f"Adversary steps at goal: {(adv_at_goal)}")
    print(f"Adversary steps at non-goal: {(adv_at_nongoal)}")
    print(f"Normal agent steps at goal: {(normal_at_goal)}")
    #print(f"Mean Reward over {config.n_episodes} episodes of agent 2 (Good): {mean_rewards[2]} with std: {np.std(total_rewards_2)}")
    #print(list_mean_1)
    #print(list_mean_2)
    #print(f"Mean self-other overlap over {config.n_episodes} episodes of agent 1 (Good): {np.mean(list_mean_1)} with std: {np.std(list_mean_1)}")
    #print(f"Mean self-other overlap over {config.n_episodes} episodes of agent 2 (Good): {np.mean(list_mean_2)} with std: {np.std(list_mean_2)}")
    
    env.close()
    
    adv_path = os.path.join('benchmarks','adversary')
    agent_path = os.path.join('benchmarks','agent')
    
    if not os.path.exists(adv_path):
        os.makedirs(adv_path)
        
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)
    #print(total_threshold_list)
    #total_non_goal_angle_list = np.array(total_non_goal_angle_list, dtype=object)
    total_threshold_list = np.array(total_threshold_list, dtype=object)
    total_action_list = np.array(total_action_list, dtype=object)
    if(config.eval_num):
        #np.save('benchmarks/adversary/run' + str(config.run_num) +"_"+str(config.eval_num)+ "_mean_adversary_benchmarks.npy", mean_adversary_benchmarks)
        #np.save('benchmarks/agent/run' + str(config.run_num) +"_"+str(config.eval_num) + "_mean_good_agent_benchmarks.npy", mean_good_agent_benchmarks)
        #np.save('benchmarks/agent/run' + str(config.run_num) +"_"+str(config.seed) + "_non_goal_angle_list.npy", total_non_goal_angle_list)
        #np.save('benchmarks/agent/run' + str(config.run_num) +"_"+str(config.seed) + "_threshold_list.npy", total_threshold_list)
        np.save('benchmarks/agent/run' + str(config.run_num) +"_"+str(config.seed) + "_action_list.npy", total_action_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--eval_num", default=1, required=False, type=int)
    parser.add_argument("--self_other", default=False, type=bool)
    parser.add_argument("--random", default=False, type=bool)
    parser.add_argument("--pre_trained", type=int)
    parser.add_argument("--self_other_strategy", default=False, type=bool)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--year", default=2024, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")

    config = parser.parse_args()

    run(config)
