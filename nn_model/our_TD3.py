import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, curr_state, action, next_state, reward, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = curr_state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        return (
            T.FloatTensor(self.state_memory[batch]).to(device),
            T.FloatTensor(self.action_memory[batch]).to(device),
            T.FloatTensor(self.new_state_memory[batch]).to(device),
            T.FloatTensor(self.reward_memory[batch]).to(device),
            T.FloatTensor(self.terminal_memory[batch]).to(device)
        )


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='./save_model/TD3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.fc3 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc4 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q2 = nn.Linear(self.fc2_dims, 1)

        self.fc5 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc6 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q3 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, action):
        action_value_1 = self.fc1(T.cat([state, action], dim=1))
        action_value_1 = F.relu(action_value_1)
        action_value_1 = self.fc2(action_value_1)
        action_value_1 = F.relu(action_value_1)

        q1 = self.q1(action_value_1)

        action_value_2 = self.fc3(T.cat([state, action], dim=1))
        action_value_2 = F.relu(action_value_2)
        action_value_2 = self.fc4(action_value_2)
        action_value_2 = F.relu(action_value_2)

        q2 = self.q2(action_value_2)

        return q1, q2

    def Q1(self, state, action):
        action_value_3 = self.fc5(T.cat([state, action], dim=1))
        action_value_3 = F.relu(action_value_3)
        action_value_3 = self.fc6(action_value_3)
        action_value_3 = F.relu(action_value_3)

        q3 = self.q3(action_value_3)

        return q3

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='./save_model/TD3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class TD3:
    def __init__(self,
                 state_dim,
                 action_dim,
                 user_id,
                 batch_size=64,
                 max_action=1,
                 discount=0.99,
                 tau=0.0005,
                 policy_noise=0.1,
                 noise_clip=1.0,
                 max_size=10000,
                 policy_freq=1,
                 reward_scale=2,
                 learning_rate=3e-4):

        self.max_action = max_action
        self.max_size = max_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.state_dim = [state_dim]
        self.action_dim = action_dim
        self.memory = ReplayBuffer(max_size, self.state_dim, action_dim)

        self.actor = ActorNetwork(learning_rate, self.state_dim, n_actions=action_dim,
                                  name='actor_' + str(user_id), max_action=max_action)
        self.actor_target = ActorNetwork(learning_rate, self.state_dim, n_actions=action_dim,
                                         name='actor_target_' + str(user_id), max_action=max_action)

        self.critic = CriticNetwork(learning_rate, self.state_dim, n_actions=action_dim,
                                    name='critic_' + str(user_id))
        self.critic_target = CriticNetwork(learning_rate, self.state_dim, n_actions=action_dim,
                                           name='critic_target_' + str(user_id))

        self.scale = reward_scale
        self.total_it = 0
        self.exploration = 0
        self.exploration_total = 1000

    def reset_memory(self):
        self.exploration = 0
        self.exploration_total = 1000
        self.memory = ReplayBuffer(self.max_size, self.state_dim, self.action_dim)

    def store_transition(self, curr_state, action, next_state, reward, done):
        self.memory.store_transition(curr_state, action, next_state, reward, done)

    def select_action(self, observation):
        if self.exploration < self.exploration_total:
            self.exploration += 1
            return np.random.randint(0, 2, 200)
        else:
            state = T.Tensor(np.array([observation])).to(device)
            actions, _ = self.actor.sample_normal(state, reparameterize=False)

            return actions.cpu().detach().numpy().reshape([200])

    def train(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.memory.sample_buffer(self.batch_size)

        with T.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    T.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action_tmp, _ = self.actor_target(next_state)
            next_action = (
                    next_action_tmp + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = T.min(target_Q1, target_Q2)
            target_Q = self.scale * reward.reshape([256, 1]) + not_done.reshape([256, 1]) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            state_, _ = self.actor(state)
            actor_loss = self.critic.Q1(state, state_).mean()

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_target.load_checkpoint()
