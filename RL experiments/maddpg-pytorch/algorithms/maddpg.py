import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from scipy.stats import beta, uniform
from torch.distributions import Beta
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()
def increasing_weight(x):
    
    x0 = torch.tensor(0, dtype=torch.float32)
    x1 = torch.tensor(5000, dtype=torch.float32)
    y0 = torch.tensor(1e-4, dtype=torch.float32)
    y1 = torch.tensor(1e-2, dtype=torch.float32)

    return (y0 + ((y1 - y0) * (x - x0)) / (x1 - x0)).item()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        # Collect only the actions for each agent
        h1 = [agent.step(obs, explore=explore)[0] for agent, obs in zip(self.agents, observations)]
        h2 = [agent.step(obs, explore=explore)[1] for agent, obs in zip(self.agents, observations)]
        actions = [agent.step(obs, explore=explore)[2] for agent, obs in zip(self.agents, observations)]
        return h1, h2, actions
    
    
    def update(self, sample, agent_i, parallel=False, logger=None, self_other = False, agent_overlap_values=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        #obs, acs, rews, next_obßßßs, dones = sample
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        if(agent_i>0 and self_other):
            if(agent_overlap_values[agent_i-1].item() != 0):
                curr_agent.policy_optimizer.zero_grad()
                #agent_overlap_values_tensor ``= agent_overslap_values[:, agent_i]
                #for SOO
                #self_other_loss =  agent_overlap_values_tensor.mean()
                #self_other_loss = -1*agent_overlap_values[agent_i-1]+agent_overlap_values[1]
                #self_other_loss = agent_overlap_values[agent_i-1]
                self_other_loss = agent_overlap_values[1]
                #epsilon = 1e-25
                #random_val = torch.log(torch.rand(1).pow(10)+epsilon).requires_grad_(True)
                #self_other_loss = random_val+agent_overlap_values[1]
                #print(self_other_loss)
                
                # Ensure operations remain on the computational graph
                #alpha, beta_param = torch.tensor(5.), torch.tensor(5.)
                #beta_dist = Beta(alpha, beta_param)
                #sample = beta_dist.sample()  # This is a tensor
                
                # Scaling your sample to the desired range
                #self_other_loss = -(0.75 + (0.85 - 0.75) * sample)
                #print("SOO Loss: ", self_other_loss)
                self_other_loss.backward()
                if parallel:
                    average_gradients(curr_agent.policy)
                torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
                curr_agent.policy_optimizer.step()
        if(self_other):
            curr_agent.critic_optimizer.zero_grad()
            if self.alg_types[agent_i] == 'MADDPG':
                if self.discrete_action: # one-hot encode action
                    all_trgt_acs = [onehot_from_logits(pi(nobs)[2]) for pi, nobs in
                                    zip(self.target_policies, next_obs)]
                else:
                    all_trgt_acs = [pi(nobs)[2] for pi, nobs in zip(self.target_policies,
                                                                next_obs)]
                trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
            #else:  # DDPG
            #    if self.discrete_action:
            #        trgt_vf_in = torch.cat((next_obs[agent_i],
            #                                onehot_from_logits(
            #                                    curr_agent.target_policy(
            #                                        next_obs[agent_i])[2])),
            #                               dim=1)
            #    else:
            #        trgt_vf_in = torch.cat((next_obs[agent_i],
            #                                curr_agent.target_policy(next_obs[agent_i])[2]),
            #                               dim=1)
            target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                            curr_agent.target_critic(trgt_vf_in)[2] *
                            (1 - dones[agent_i].view(-1, 1)))
            
            if self.alg_types[agent_i] == 'MADDPG':
                vf_in = torch.cat((*obs, *acs), dim=1)
            #else:  # DDPG
            #    vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
            actual_value = curr_agent.critic(vf_in)[2]
            vf_loss = MSELoss(actual_value, target_value.detach())
            #print("vf_loss: ", vf_loss)
            vf_loss.backward()
            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
            curr_agent.critic_optimizer.step()

            curr_agent.policy_optimizer.zero_grad()

            if self.discrete_action:
                # Forward pass as if onehot (hard=True) but backprop through a differentiable
                # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
                # through discrete categorical samples, but I'm not sure if that is
                # correct since it removes the assumption of a deterministic policy for
                # DDPG. Regardless, discrete policies don't seem to learn properly without it.
                curr_pol_out = curr_agent.policy(obs[agent_i])[2]
                curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
            else:
                curr_pol_out = curr_agent.policy(obs[agent_i])[2]
                curr_pol_vf_in = curr_pol_out
            if self.alg_types[agent_i] == 'MADDPG':
                all_pol_acs = []
                for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                    if i == agent_i:
                        all_pol_acs.append(curr_pol_vf_in)
                    elif self.discrete_action:
                        all_pol_acs.append(onehot_from_logits(pi(ob)[2]))
                    else:
                        all_pol_acs.append(pi(ob)[2])
                vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                                dim=1)
            pol_loss = -curr_agent.critic(vf_in)[2].mean()
            pol_loss += (curr_pol_out**2).mean() * 1e-3
            if(agent_i>5): 
               #self_other_loss = agent_overlap_values[agent_i]
               epsilon = 1e-25
               self_other_loss = torch.log(torch.rand(1).pow(10)+epsilon).requires_grad_(True)
                # Clip self_other_loss to be between -2 and 2
            #   self_other_loss_clipped = torch.clamp(self_other_loss, min=-4, max=4)
                #if(self_other_loss.item()<-10): print(self_other_loss)
                #pol_loss = pol_loss + 1e-6 * self_other_loss_clipped
               pol_loss = pol_loss  + 1e-4*self_other_loss
            #print("pol_loss: ", pol_loss)
            pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
        
        
        if logger is not None and self_other:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1
    
    def update_specific_target(self, agent_idx):
        """
        Update specific target networks (called after normal updates have been
        performed for the agents)
        """
        soft_update(self.agents[0].target_critic, self.agents[0].critic, self.tau)
        soft_update(self.agents[0].target_critic, self.agents[0].critic, self.tau)
        
        soft_update(self.agents[agent_idx].target_critic, self.agents[agent_idx].critic, self.tau)
        soft_update(self.agents[agent_idx].target_critic, self.agents[agent_idx].critic, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                #get_shape = lambda x: x.n
                get_shape = lambda x: x.shape[0]
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance