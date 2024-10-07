import argparse
import torch
import os
import sys
import numpy as np
from pathlib import Path
import statistics, random, math
import torch.nn.functional as F
from scipy.stats import beta, uniform
from torch.autograd import Variable
from gym.spaces import Box
from collections import Counter
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
import gc
from scipy.stats import truncnorm

USE_CUDA = False  # torch.cuda.is_available()

def old_initialise_adversary(normal_agent, current_adversary, sight_radius):
        wall_width = 0.1
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, sight_radius)
        x_offset = distance * math.cos(angle)
        y_offset = distance * math.sin(angle)

        # Calculate the new position, ensuring it's within the bounds of -1 and +1
        new_x = np.clip(normal_agent[0] + x_offset, -1+wall_width, 1-wall_width)
        new_y = np.clip(normal_agent[1] + y_offset, -1+wall_width, 1-wall_width)
        
        current_adversary[0] = new_x
        current_adversary[1] = new_y
        
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
    
    if strategy == 'mixed':
        # Mixed strategy 1 (consider both the other good agent and the adversary as other)
        torch_obs[1][0, 7] = -torch_obs[1][0, 7]
        torch_obs[1][0, 8] = -torch_obs[1][0, 8]
        torch_obs[2][0, 7] = -torch_obs[2][0, 7]
        torch_obs[2][0, 8] = -torch_obs[2][0, 8]
    
    elif strategy == 'adversary_only':
        # Specifically target the adversary (only consider the adversary as the other)
        torch_obs[1][0, 6] = -torch_obs[1][0, 6]
        torch_obs[1][0, 7] = -torch_obs[1][0, 7]
        torch_obs[2][0, 6] = -torch_obs[2][0, 6]
        torch_obs[2][0, 7] = -torch_obs[2][0, 7]

    elif strategy == 'total':
        # Total strategy where one considers both the adversary and the other normal agent as the 'other'
        
        # The adversary 
        torch_obs[1][0, 6] = -torch_obs[1][0, 6]
        torch_obs[1][0, 7] = -torch_obs[1][0, 7]
        torch_obs[2][0, 6] = -torch_obs[2][0, 6]
        torch_obs[2][0, 7] = -torch_obs[2][0, 7]
        # The other normal agent
        torch_obs[1][0, 8] = -torch_obs[1][0, 8]
        torch_obs[1][0, 9] = -torch_obs[1][0, 9]
        torch_obs[2][0, 8] = -torch_obs[2][0, 8]
        torch_obs[2][0, 9] = -torch_obs[2][0, 9]
        
    elif strategy == 'full_adversary':
        # Specifically targeting the adversary (and adjusting all relative positions accordingly)

        if(torch.equal(torch_obs[1][0, 0:2],torch_obs[1][0, 2:4])):
            goal = "landmark_1"
        elif(torch.equal(torch_obs[1][0, 0:2],torch_obs[1][0, 4:6])):
            goal = "landmark_2"
            
        if(goal=="landmark_1"):
            torch_obs[1] = torch.cat([torch_obs[0][0, 0:2], torch_obs[0][0]]).unsqueeze(0)  # Selecting first two elements from torch_obs[1]
             # Concatenating specific elements from torch_obs[2] and torch_obs[0]
            torch_obs[2] = torch.cat([torch_obs[0][0, 0:2], torch_obs[0][0]]).unsqueeze(0)  # Selecting first two elements from torch_obs[2]
            
        elif(goal=="landmark_2"):
            torch_obs[1] = torch.cat([torch_obs[0][0, 2:4], torch_obs[0][0]]).unsqueeze(0)  # Selecting first two elements from torch_obs[1]
            # Concatenating specific elements from torch_obs[2] and torch_obs[0]
            torch_obs[2] = torch.cat([torch_obs[0][0, 2:4], torch_obs[0][0]]).unsqueeze(0)  # Selecting first two elements from torch_obs[2]
    elif strategy=="create_other_observation":
            
            sight_radius = 2.5*0.05
            if(agent_idx==1 or agent_idx==3):
                old_initialise_adversary(positions[1], positions[0],sight_radius)

                agent1_adv_oobs = positions[1]-positions[0]
                
                torch_obs[1][0, 6] = agent1_adv_oobs[0]
                torch_obs[1][0, 7] = agent1_adv_oobs[1]

            if(agent_idx==2 or agent_idx==3):
                old_initialise_adversary(positions[2], positions[0],sight_radius)
                agent2_adv_obs = positions[2]-positions[0]
                
                torch_obs[2][0, 6] = agent2_adv_obs[0] 
                torch_obs[2][0, 7] = agent2_adv_obs[1]
    elif strategy=="new_create_other_observation":
        #torch.autograd.set_detect_anomaly(True)
        sight_radius = 2.5*0.05
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
            
    elif strategy=="create_self_observation":
        if(agent_idx==1 or agent_idx==3):
                # adversary and normal agent positions
                torch_obs[1][0, 6] = torch.tensor(0)
                torch_obs[1][0, 7] = torch.tensor(0)
                torch_obs[1][0, 10] = torch.tensor(0)
                torch_obs[1][0, 11] = torch.tensor(0)
                # relative velocities
                torch_obs[1][0, 8] = torch.tensor(0)
                torch_obs[1][0, 9] = torch.tensor(0)
                torch_obs[1][0, 12] = torch.tensor(0)
                torch_obs[1][0, 13] = torch.tensor(0)
        if(agent_idx==2 or agent_idx==3):
                # adversary and normal agent positions
                torch_obs[2][0, 6] = torch.tensor(0)
                torch_obs[2][0, 7] = torch.tensor(0)
                torch_obs[2][0, 10] = torch.tensor(0)
                torch_obs[2][0, 11] = torch.tensor(0)
                # relative velocities
                torch_obs[2][0, 8] = torch.tensor(0)
                torch_obs[2][0, 9] = torch.tensor(0)
                torch_obs[2][0, 12] = torch.tensor(0)
                torch_obs[2][0, 13] = torch.tensor(0)
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
            
            
            
            
        
        
def about_self(torch_obs):
    agent_nr = 0
    if((torch_obs[1][0, 6].item() == 0 and torch_obs[1][0, 7].item() == 0) and (torch_obs[1][0, 8].item() == 0 and torch_obs[1][0,9].item() == 0)): agent_nr += 1
    if((torch_obs[2][0, 6].item() == 0 and torch_obs[2][0, 7].item() == 0) and (torch_obs[2][0, 8].item() == 0 and torch_obs[2][0,9].item() == 0)): agent_nr += 2
    return agent_nr

def about_other(torch_obs):
    #here by other we mean the adversary
    agent_nr = 0 
    if((torch_obs[1][0, 6].item() != 0 and torch_obs[1][0, 7].item() != 0) and (torch_obs[1][0, 8].item() == 0 and torch_obs[1][0,9].item() == 0)): agent_nr += 1
    if((torch_obs[2][0, 6].item() != 0 and torch_obs[2][0, 7].item() != 0) and (torch_obs[2][0, 8].item() == 0 and torch_obs[2][0,9].item() == 0)): agent_nr += 2
    return agent_nr

def about_other_vel(torch_obs):
    #here by other we mean the adversary
    agent_nr = 0 
    if((torch_obs[1][0, 6].item() != 0 and torch_obs[1][0, 7].item() != 0) and (torch_obs[1][0, 10].item() == 0 and torch_obs[1][0,11].item() == 0)): agent_nr += 1
    if((torch_obs[2][0, 6].item() != 0 and torch_obs[2][0, 7].item() != 0) and (torch_obs[2][0, 10].item() == 0 and torch_obs[2][0,11].item() == 0)): agent_nr += 2
    return agent_nr

def old_about_self_vel(torch_obs):
    #here by self we mean normal agent 
    agent_nr = 0 
    if((torch_obs[1][0, 6].item() == 0 and torch_obs[1][0, 7].item() == 0) and (torch_obs[1][0, 10].item() == 0 and torch_obs[1][0,11].item() == 0)): agent_nr += 1
    if((torch_obs[2][0, 6].item() == 0 and torch_obs[2][0, 7].item() == 0) and (torch_obs[2][0, 10].item() == 0 and torch_obs[2][0,11].item() == 0)): agent_nr += 2
    return agent_nr

def about_self_vel(torch_obs):
    #here by self we mean normal agent 
    agent_nr = 0
    if(torch_obs[1][0, 6].item() == 0 and torch_obs[1][0, 7].item() == 0): agent_nr += 1
    #if((torch_obs[2][0, 8].item() == 0 and torch_obs[2][0, 9].item() == 0) and (torch_obs[2][0, 12].item() == 0 and torch_obs[2][0,13].item() == 0)): agent_nr += 2
    return agent_nr


def old_mean_squared_error(activation_X, activation_Y):
    # Ensure the inputs are numpy arrays for element-wise operations
    activation_X = np.array(activation_X.detach().numpy())
    activation_Y = np.array(activation_Y.detach().numpy())
    
    # Compute the squared differences between corresponding elements
    squared_differences = (activation_X - activation_Y) ** 2
    
    # Calculate the mean of these squared differences
    mse = np.mean(squared_differences)

    # Ensure mse is positive to avoid log(0)
    epsilon = 1e-25
    log_mse = np.log(mse+epsilon)
    
    return log_mse

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

def new_mean_squared_error(activation_X, activation_Y, alpha=2.0, beta=0.5):
    # Convert inputs to tensors if they are not already, ensuring they are of the same type
    if not isinstance(activation_X, torch.Tensor):
        activation_X = torch.tensor(activation_X, dtype=torch.float32)
    if not isinstance(activation_Y, torch.Tensor):
        activation_Y = torch.tensor(activation_Y, dtype=torch.float32)
    
    # Ensure alpha and beta are tensors of the same dtype and device as activation_X
    alpha_tensor = torch.full_like(activation_X, alpha)
    beta_tensor = torch.full_like(activation_X, beta)
    
    # Compute the squared differences
    squared_differences = (activation_X - activation_Y) ** 2
    
    # Determine the weights: alpha_tensor for activation_Y < activation_X, beta_tensor otherwise
    weights = torch.where(activation_Y < activation_X, alpha_tensor, beta_tensor)
    
    # Apply the weights
    weighted_squared_differences = weights * squared_differences
    
    # Calculate the mean of these weighted squared differences
    weighted_mse = torch.mean(weighted_squared_differences)
    
    # Adding a small epsilon to the weighted_mse before taking log to prevent log(0)
    epsilon = 1e-25
    log_weighted_mse = torch.log(weighted_mse + epsilon)
    
    return log_weighted_mse

def extremely_biased_random(min_value=0, max_value=2800, bias=800):
    biased_float = random.random() ** bias
    scaled_value = min_value + (max_value - min_value) * biased_float
    return scaled_value


def mean_absolute_error(X, Y):
    """
    Calculate the Mean Absolute Error (MAE) between two matrices.
    
    Parameters:
    - X: numpy array of shape (n_rows, n_columns)
    - Y: numpy array of the same shape as X
    
    Returns:
    - mae: The MAE between matrices X and Y.
    """
    mae = np.mean(np.abs(X - Y))
    return mae

def custom_distance_function(matrix1, matrix2):
    """
    Calculate a scalar value by taking the product of (1 - fractional part of the positive differences)
    and the reciprocal of the positive differences between corresponding elements in matrix1 and matrix2,
    then summing them up. This method incentivizes smaller magnitudes and fractional parts of differences.
    
    Parameters:
    - matrix1: First input matrix (NumPy array).
    - matrix2: Second input matrix (NumPy array).
    
    Returns:
    - float: The scalar value representing the sum of the adjusted reciprocals of positive differences.
    """
    # Calculate the difference between the two matrices
    differences = matrix1 - matrix2
    
    # Filter out differences less than 0 by setting them to a high number which will later be turned to nearly 0 in the reciprocal calculation
    positive_differences = np.where(differences < 0, sys.maxsize, differences)
    
    # Calculate the fractional part of each positive difference
    fractional_parts = positive_differences - np.floor(positive_differences)
    
    # Calculate the incentive for small fractional parts
    fractional_incentive = 1 - fractional_parts
    
    # Adjust differences: set differences < 1 to 1
    adjusted_differences = np.where(positive_differences < 1, 1, positive_differences)
    
    # Calculate the reciprocal of positive differences
    reciprocal_values = 1 / adjusted_differences
    
    #Set the negative differences to 0
    clean_reciprocal_values = np.where(reciprocal_values==1/sys.maxsize,0,reciprocal_values)
    
    # Multiply the reciprocal values by the fractional incentive
    adjusted_values = clean_reciprocal_values * fractional_incentive
    
    # Give double weight to whole number part overlap 
    weighted_values = 2*clean_reciprocal_values+adjusted_values

    # Sum up the adjusted values
    self_other_overlap = np.sum(weighted_values)
    
    return self_other_overlap


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

def normalise(value, func):
    def modified_sigmoid(x, k=5, x0=20):
        return 1 / (1 + np.exp(-k * (x - x0)))
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def inverted_exponential(x, k=0.005):
        return 1 - np.exp(-k * x)
    
    def random_control(x, range):
        #for non-specific random reward
        
        #for specific random reward
        if(x!=0): return random.uniform(range[0], range[1])
        else: return 0

    if func == "tanh":
        return np.tanh(value)
    if func == "modified_sigmoid":
        return modified_sigmoid(value)
    if func == "inverted_exponential":
        return inverted_exponential(value)
    if func == "random_control":
        return random_control(value, [0,1])
    if func == "sigmoid":
        return sigmoid(value)
 
def custom_adjusted_control(value, mean, std, lower_bound=0, upper_bound=np.inf):
    """
    Generate a random number from a truncated normal distribution.

    Parameters:
    - mean: The mean of the distribution.
    - std: The standard deviation of the distribution.
    - lower_bound: The lower bound of the distribution (default is 0).
    - upper_bound: The upper bound of the distribution (default is np.inf).

    Returns:
    - A random number from the truncated normal distribution.
    """
    if(value==0 or std==0 or upper_bound==0):
        return 0
    else:
        # Calculate the boundaries in terms of standard deviations from the mean
        a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
        
        # Generate a random number from the truncated normal distribution
        random_number = truncnorm(a, b, loc=mean, scale=std).rvs()
        
        return random_number

def adjusted_control(value, mean, std):
    if(value!=0 and mean!=0 and std!=0):
        # Calculate alpha and beta from desired mean and std
        a = ((1 - mean) / std**2 - 1 / mean) * mean**2
        b = a * (1 / mean - 1)
        
        # Validate parameters
        if a <= 0 or b <= 0:
            print("Cannot achieve the desired mean and std with a beta distribution in [0,1]. Falling back to uniform distribution in [0,1].")
            # Fallback to uniform distribution
            random_number = uniform.rvs(0, 1)
        else:
            # Generate a single random number from the beta distribution
            random_number = beta.rvs(a, b)
        return random_number
    else:
        return 0

def calculate_kl_divergence(pretrained_actions, current_actions):
    # Assuming both inputs are already tensors. If not, they should be converted to tensors.
    pretrained_probs = F.softmax(pretrained_actions, dim=-1)
    current_probs = F.softmax(current_actions, dim=-1)
    
    # Calculate KL divergence. Note that the first argument should be log probabilities.
    kl_divergence = F.kl_div(pretrained_probs.log(), current_probs, reduction='batchmean')
    return kl_divergence.item()

def old_pca_directions(X):
    X_mean_centered = X - torch.mean(X, dim=0)
    covariance_matrix = torch.mm(X_mean_centered.t(), X_mean_centered) / (X.size(0) - 1)
    eigenvalues, eigenvectors = torch.symeig(covariance_matrix, eigenvectors=True)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvectors

def pca_directions(X):
    #print("X: ", X)
    X_mean_centered = X - torch.mean(X, dim=0)
    #print("X_mean_centered: ", X_mean_centered )
    covariance_matrix = torch.mm(X_mean_centered.t(), X_mean_centered) / (X.size(0) - 1)
    # Ensure the matrix is symmetric
    covariance_matrix = (covariance_matrix + covariance_matrix.t()) / 2
    # Regularization to improve numerical stability
    regularization_term = 1e-4
    I = torch.eye(covariance_matrix.size(0), dtype=covariance_matrix.dtype, device=covariance_matrix.device)
    covariance_matrix += I * regularization_term
    #print("covareince matrix: ", covariance_matrix)
    # Use a more stable eigen solver if possible
    eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvectors

def pca_overlap(X1, X2):
    # Calculate PCA directions for each matrix
    pca1 = pca_directions(X1)
    pca2 = pca_directions(X2)
    
    # Calculate cosine similarity for corresponding eigenvectors
    # Normalize the eigenvectors as they should theoretically already be but to ensure numerical stability
    pca1_normalized = torch.nn.functional.normalize(pca1, dim=0)
    pca2_normalized = torch.nn.functional.normalize(pca2, dim=0)
    
    # Calculate absolute cosine similarities
    cos_similarities = torch.abs(torch.sum(pca1_normalized * pca2_normalized, dim=0))
    
    # Average the cosine similarities across all principal components
    average_overlap = torch.mean(cos_similarities)
    return average_overlap
    
def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action, benchmark=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)]) if n_rollout_threads == 1 else SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def print_grad(grad):
    print(grad)

USE_CUDA = False  # torch.cuda.is_available()

def run(config):
    #torch.autograd.set_detect_anomaly(True)
    model_dir = Path('./models') / config.env_id / config.model_name
    curr_run = 'run1' if not model_dir.exists() else 'run%i' % (max([int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')], default=0) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.discrete_action)
    maddpg = MADDPG.init_from_save(Path('./models') / config.env_id / config.model_name / ('run%i' % config.load) / 'model.pt') if config.load else MADDPG.init_from_env(env, agent_alg=config.agent_alg, adversary_alg=config.adversary_alg, tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim)
    pretrained_maddpg = MADDPG.init_from_save(Path('./models') / config.env_id / config.model_name / ('run%i' % config.pre_trained) / 'model.pt') if config.pre_trained else None

    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents, [obsp.shape[0] for obsp in env.observation_space], [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space])
    t = 0
    total_self_other_1 = []
    total_self_other_2 = []
    total_adv_in_sight = []
    mean_adversary_benchmarks, mean_good_agent_benchmarks = [], [[], []]
    mean_self_other_1, mean_self_other_2 = [], []
    std_self_other_1, std_self_other_2 = [], []
    total_deceptive_rew = []
    total_adv_rew = []
    total_pos_rew = []
    
    total_reached_goal = 0
    
    total_reached_nongoal = 0
    
    total_deceptive = 0
    
    total_dif = []
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes))
        obs = env.reset()
        
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()
        ep_adv_in_sight = 0
        ep_mean_self_other_1, ep_mean_self_other_2 = [], []
        ep_benchmark_adversary, ep_benchmark_good_agents = 0, [0] * 2
        adversary_benchmarks, good_agent_benchmarks = [], [[], []]
        distance_to_goal, distance_to_landmark = [0,0], [0,0]
        ep_deceptive_rew, ep_adv_rew, ep_pos_rew = [], [], []
        
        if config.pre_trained:
            pretrained_maddpg.prep_rollouts(device='cpu')
            pretrained_maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
            pretrained_maddpg.reset_noise()
        #if(ep_i%2==0):
        #    env.set_deceptive(True)
        #else:
        #    env.set_deceptive(False)
        #env.set_deceptive(True)
        #env.set_deceptive(True)
        mean_kl_1 = []
        mean_kl_2 = []
        for t_i in range(config.episode_length):
            
            #print(range(maddpg.nagents))
            #print("obs: ", obs)
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(maddpg.nagents)]
            #torch_obs_modified = [Variable(obs_tensor.data.clone(), requires_grad=False) for obs_tensor in torch_obs]
            #print("torch_obs: ", torch_obs)
            #print("torch_obs_modified: ", torch_obs_modified)
           
            #other_in_sight = about_other_vel(torch_obs)
            #if(other_in_sight==1): 
            #    ep_adv_in_sight += 1
            #if(other_in_sight==2): 
            #    ep_adv_in_sight += 1
            #if(other_in_sight==3): 
            #    ep_adv_in_sight += 2
            #torch.autograd.set_detect_anomaly(True)
            #print(torch_obs)
            h1, h2, torch_agent_actions = maddpg.step(torch_obs, explore=True)
            #print(torch_agent_actions)
            #h1, h2, torch_agent_actions_2 = maddpg.step(torch_obs_modified, explore=True)

            #print("torch_agent_actions:", torch_agent_actions)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            #print(agent_actions)
            #print(agent_actions[1][0])
            #print("torch_agent_actions:", torch_agent_actions)
            #agent_actions_2 = [ac.data.numpy() for ac in torch_agent_actions_2]

            if config.pre_trained:
                pretrained_h1, pretrained_h2, pretrained_torch_agent_actions = pretrained_maddpg.step(torch_obs, explore=False)
                #kl_div_0 = calculate_kl_divergence(pretrained_torch_agent_actions[0], torch_agent_actions[0])

                #kl_div_1 = calculate_kl_divergence(pretrained_torch_agent_actions[1], torch_agent_actions[1])
                #mean_kl_1.append(kl_div_1)

                #kl_div_2 = calculate_kl_divergence(pretrained_torch_agent_actions[2], torch_agent_actions[2])
                #mean_kl_2.append(kl_div_2)

                
                pretrained_agent_actions = [ac.data.numpy() for ac in pretrained_torch_agent_actions]
                #print(pretrained_agent_actions)
                #print(pretrained_agent_actions[1][0])
                #print(pretrained_torch_agent_actions)
                #print(pretrained_torch_agent_actions[1][0])
            

            for agent_idx, agent in enumerate(env.envs[0].world.agents):
                        benchmark_data = env.envs[0]._get_info(agent)
                        if agent.adversary:
                                ep_benchmark_adversary += benchmark_data
                        else:
                                distance_to_goal[agent_idx] = next((num for num, count in Counter(benchmark_data).items() if count > 1), 0) 
                                #repeats twice as it is both a landmark and a goal
                                distance_to_landmark[agent_idx] = next((num for num, count in Counter(benchmark_data).items() if count == 1), 0) #distance to non-goal landmark
                                    
                        # Find index of the agent closest to the goal
                        closest_to_goal_idx = 0 if distance_to_goal[0] < distance_to_goal[1] else 1
                        # Find index of the other agent (the one not closest to the goal)
                        other_agent_idx = 1 - closest_to_goal_idx
                        ep_benchmark_good_agents[0] += distance_to_goal[1]
                        ep_benchmark_good_agents[1] += distance_to_landmark[1]
            
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            #print(actions)
            #print(actions[0][1])
            #actions_2 = [[ac[i] for ac in agent_actions_2] for i in range(config.n_rollout_threads)]


            next_obs, rewards, dones, infos = env.step(actions)
            
            
            
            ep_deceptive_rew.append(rewards[0][1])
            ep_adv_rew.append(-rewards[0][0])
            ep_pos_rew.append(rewards[0][1]+rewards[0][0])
            


            positions  = env.envs[0]._get_info(t_i, True)
            if(t_i==config.episode_length-1):
                ep_adv_in_sight = positions[2]
                #total_reached_goal += positions[4]
                #total_reached_nongoal += positions[5]
                #total_deceptive += positions[6]
            
            #old_self_other_overlap_1, old_self_other_overlap_2 = off_self_other_overlap(h1, h2, torch_obs, maddpg, positions, strategy='create_other_observation')
            #self_other_overlap_1, self_other_overlap_2 = self_other_overlap(torch_agent_actions[1], h1, h2, torch_obs, maddpg, positions, strategy='new_create_other_observation')
            self_other_overlap_1, self_other_ovlerap_2 = torch.tensor(0), torch.tensor(0)
            #if(type(dif) is not int):
                #print(dif)
            #    total_dif.append(dif)
                #print("Mean: ", torch.mean(torch.stack(total_dif), dim=0))
                
            if(config.pre_trained):
                self_other_overlap_2 = mean_squared_error(pretrained_torch_agent_actions[1][0], torch_agent_actions[1][0])
            else:
                self_other_overlap_2 = torch.tensor(0)
            #self_other_overlap_1 = torch.tensor(self_other_overlap_1)
            #self_other_overlap_2 = torch.tensor(self_other_overlap_2)
            #self_other_overlap_1, self_other_overlap_2 = 0, 0
            #elf_other_overlap_1.register_hook(print_grad)
            #print("old_self_other_overlap_1: ", old_self_other_overlap_1)
            #print("new_self_other_overlap_1: ", self_other_overlap_1)
            #print("old_self_other_overlap_2: ", old_self_other_overlap_2)
            #print("new_self_other_overlap_1: ", self_other_overlap_2)
            if(self_other_overlap_1!=0): total_self_other_1.append(self_other_overlap_1)
            if(self_other_overlap_2!=0): total_self_other_2.append(self_other_overlap_2)
            if(self_other_overlap_1!=0): ep_mean_self_other_1.append(self_other_overlap_1)
            if(self_other_overlap_2!=0): ep_mean_self_other_2.append(self_other_overlap_2)
            

            
            if config.self_other:

                rewards[0][1] += normalise(self_other_overlap_1, "inverted_exponential")
                rewards[0][2] += normalise(self_other_overlap_2, "inverted_exponential")
                    

                
                if ep_i % 500 == 0 and ep_mean_self_other_1!= [] and ep_mean_self_other_2!=[]:
                    
                    print(f"Mean self-other overlap of agent 1 (Good): {np.mean(ep_mean_self_other_1)} with std: {np.std(ep_mean_self_other_1)}")
                    print(f"Mean self-other overlap of agent 2 (Good): {np.mean(ep_mean_self_other_2)} with std: {np.std(ep_mean_self_other_2)}")
            if config.random:

                alpha, beta_param = 5, 5 

                
                rewards[0][1] += 0.75 + (0.85 - 0.75) * beta.rvs(alpha, beta_param)
                rewards[0][2] += 0.75 + (0.85 - 0.75) * beta.rvs(alpha, beta_param)
                
                
                if ep_i % 500 == 0 and ep_mean_self_other_1!= [] and ep_mean_self_other_2!=[]:
                    print("Mean: ", torch.mean(torch.stack(total_dif), dim=0))
                    print(f"Mean self-other overlap of agent 1 (Good): {np.mean(ep_mean_self_other_1)} with std: {np.std(ep_mean_self_other_1)}")
                    print(f"Mean self-other overlap of agent 2 (Good): {np.mean(ep_mean_self_other_2)} with std: {np.std(ep_mean_self_other_2)}")
            #For SOO
            # Convert the integer 0 to a tensor
            #zero_tensor = torch.tensor(0, dtype=torch.float32, device=self_other_overlap_1.device)
            #zero_tensor = torch.tensor(0)
            #agent_overlap_values = [zero_tensor, torch.tensor(self_other_overlap_1), torch.tensor(self_other_overlap_2)]
            agent_overlap_values = [self_other_overlap_1, self_other_overlap_2]
            #print("agent_overlap_Values: ", agent_overlap_values)
            # Since all elements are now tensors, torch.stack should work without error
            #For Random control
            #agent_overlap_values = [0, np.log(max(1e-100,random.random()**10)), np.log(max(1e-100,random.random()**10))]
            if ep_i%250 == 0: print("agent_overlap_values: ", agent_overlap_values)
            #print("old_mse: ", old_mean_squared_error(h2[1],h2[1]))
            #print("new_mse: ", mean_squared_error(h2[1],h2[1]))
            
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            
            #In case you want to update at every inference step/at random
            update_probability = -1
            if(random.random()<update_probability):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = obs, agent_actions, rewards, next_obs, dones, agent_overlap_values
                        maddpg.update(sample, a_i, logger=logger, self_other=True)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
            obs = next_obs
            #del agent_overlap_values
            #gc.collect()
            
            t += config.n_rollout_threads


            if(not config.eval):
                if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                    if USE_CUDA:
                        maddpg.prep_training(device='gpu')
                    else:
                        maddpg.prep_training(device='cpu')
                    for u_i in range(config.n_rollout_threads):
                        for a_i in range(maddpg.nagents):
                            #sample_list = list(replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA))
                            #print("sample: ", sample)
                            #sample_list[5] = agent_overlap_values
                            #sample = tuple(sample_list)
                            sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                            maddpg.update(sample, a_i, logger=logger, self_other=True, agent_overlap_values=agent_overlap_values)
                        maddpg.update_all_targets()
                    maddpg.prep_rollouts(device='cpu')
            #else:
            #    if(other_in_sight!=0):
            #        if (len(replay_buffer) >= config.batch_size and
            #            (t % config.steps_per_update) < config.n_rollout_threads):
            #            if USE_CUDA:
            #                maddpg.prep_training(device='gpu')
            #            else:
            #                maddpg.prep_training(device='cpu')
            #            for u_i in range(config.n_rollout_threads):
            #                if(other_in_sight==3):
            #                    print("Both normal agents")
            #                    for a_i in range(maddpg.nagents):
            #                        sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
            #                        maddpg.update(sample, a_i, logger=logger)
            #                    maddpg.update_all_targets()
            #                else:
            #                    print("Only agent ", other_in_sight, " and the adversary.")
            #                    sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
            #                    maddpg.update(sample, 0, logger=logger)
            #                    maddpg.update(sample, other_in_sight, logger=logger)
            #                    maddpg.update_specific_target(other_in_sight)
            #                    
            #            maddpg.prep_rollouts(device='cpu')       

        
        # Logging and saving logic
        if len(ep_mean_self_other_1) > 0:
            mean_self_other_1.append(torch.mean(torch.stack(ep_mean_self_other_1)).detach().numpy())
            std_self_other_1.append(torch.std(torch.stack(ep_mean_self_other_1)).detach().numpy())
        if len(ep_mean_self_other_2) > 0:
            mean_self_other_2.append(torch.mean(torch.stack(ep_mean_self_other_2)).detach().numpy())
            std_self_other_2.append(torch.std(torch.stack(ep_mean_self_other_2)).detach().numpy())
        total_adv_in_sight.append(ep_adv_in_sight)
        adversary_benchmarks.append(ep_benchmark_adversary)
        mean_adversary_benchmarks.append(statistics.mean(adversary_benchmarks))
        for i in range(len(ep_benchmark_good_agents)):
            good_agent_benchmarks[i].append(ep_benchmark_good_agents[i])
            mean_good_agent_benchmarks[i].append(statistics.mean(good_agent_benchmarks[i]))
        
        total_deceptive_rew.append(np.mean(ep_deceptive_rew))
        total_adv_rew.append(np.mean(ep_adv_rew))
        total_pos_rew.append(np.mean(ep_pos_rew))
        
        ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    # Final saving of benchmarks and closing logger
    env.close()
    
    adv_path = os.path.join('benchmarks','adversary')
    agent_path = os.path.join('benchmarks','agent')
    soo_path = os.path.join('benchmarks', 'self-other')
    
    if not os.path.exists(adv_path):
        os.makedirs(adv_path)
        
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)
        
    if not os.path.exists(soo_path):
        os.makedirs(soo_path)
        
    np.save('benchmarks/adversary/' + str(curr_run) + "_mean_adversary_benchmarks.npy", mean_adversary_benchmarks)
    np.save('benchmarks/agent/' + str(curr_run) + "_mean_good_agent_benchmarks.npy", mean_good_agent_benchmarks)
    np.save('benchmarks/agent/' + str(curr_run) + "_adv_in_sight.npy", total_adv_in_sight)
    np.save('benchmarks/agent/' + str(curr_run) + "_total_deceptive_rew.npy", total_deceptive_rew)
    np.save('benchmarks/agent/' + str(curr_run) + "_total_adv_rew.npy", total_adv_rew)
    np.save('benchmarks/agent/' + str(curr_run) + "_total_pos_rew.npy", total_pos_rew)
    
    #print("Mean: ", torch.mean(torch.stack(total_dif), dim=0))
    #print("Number of episodes normal agent reached goal landmark: ", total_reached_goal)
    #print("Number of episodes adversary reached non-goal landmark: ", total_reached_nongoal)
    #print("Number of episodes adversary was deceived: ", total_deceptive)
    
    if(config.self_other or config.random):
        print("Agent 1")
        print("mean_1: " + str(torch.mean(torch.stack(total_self_other_1))))
        print("std_1: " + str(torch.std(torch.stack(total_self_other_1))))
        print("min_1: " + str(torch.min(total_self_other_1)))
        print("max_1: " + str(torch.max(total_self_other_1)))
        print("Agent 2")
        print("mean_2: " + str(torch.mean(torch.stack(total_self_other_2))))
        print("std_2: " + str(torch.std(torch.stack(total_self_other_2))))
        print("min_2: " + str(torch.min(torch.stack(total_self_other_2))))
        print("max_2: " + str(torch.max(torch.stack(total_self_other_2))))
        np.save('benchmarks/self-other/' + str(curr_run) + "_mean_self_other_1_benchmarks.npy", mean_self_other_1)
        np.save('benchmarks/self-other/' + str(curr_run) + "_mean_self_other_2_benchmarks.npy", mean_self_other_2)
        np.save('benchmarks/self-other/' + str(curr_run) + "_std_self_other_1_benchmarks.npy", std_self_other_1)
        np.save('benchmarks/self-other/' + str(curr_run) + "_std_self_other_2_benchmarks.npy", std_self_other_2)
    else:
        print("Agent 1")
        print("mean_1: " + str(torch.mean(torch.stack(total_self_other_1))))
        #print("std_1: " + str(torch.std(torch.stack(total_self_other_1))))
        #print("min_1: " + str(torch.min(torch.stack(total_self_other_1))))
        #print("max_1: " + str(torch.max(torch.stack(total_self_other_1))))
        #print("Agent 2")
        #print("mean_2: " + str(torch.mean(torch.stack(total_self_other_2))))
        #print("std_2: " + str(torch.std(torch.stack(total_self_other_2))))
        #print("min_2: " + str(torch.min(torch.stack(total_self_other_2))))
        #print("max_2: " + str(torch.max(torch.stack(total_self_other_2))))
        np.save('benchmarks/self-other/' + str(curr_run) + "_mean_self_other_1_benchmarks.npy", mean_self_other_1)
        #np.save('benchmarks/self-other/' + str(curr_run) + "_mean_self_other_2_benchmarks.npy", mean_self_other_2)
        np.save('benchmarks/self-other/' + str(curr_run) + "_std_self_other_1_benchmarks.npy", std_self_other_1)
        #np.save('benchmarks/self-other/' + str(curr_run) + "_std_self_other_2_benchmarks.npy", std_self_other_2)
    #print(f"Mean self-other overlap over {config.n_episodes} episodes of agent 1 (Good): {np.mean(list_mean_1)} with std: {np.std(list_mean_1)}")
    #print(f"Mean self-other overlap over {config.n_episodes} episodes of agent 2 (Good): {np.mean(list_mean_2)} with std: {np.std(list_mean_2)}")
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--load", type=int)
    parser.add_argument("--mean_1", type=float)
    parser.add_argument("--std_1", type=float)
    parser.add_argument("--min_1", type=float)
    parser.add_argument("--max_1", type=float)
    parser.add_argument("--mean_2", type=float)
    parser.add_argument("--std_2", type=float)
    parser.add_argument("--min_2", type=float)
    parser.add_argument("--max_2", type=float)
    parser.add_argument("--pre_trained", type=int)
    parser.add_argument("--self_other", default=False, type=bool)
    parser.add_argument("--random", default=False, type=bool)
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--year", default=False, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int) #used to be 100
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training") #used to be 
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float) #used to be 0.01
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
