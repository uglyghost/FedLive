import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_model.replay_memory import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            user_id,
            batch_size=64,
            max_action=1,
            discount=0.99,
            tau=0.0005,
            policy_noise=0.1,
            noise_clip=1.0,
            policy_freq=1,
            reward_scale=2
    ):

        self.user_id = user_id
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_target_optimizer = torch.optim.Adam(self.actor_target.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_target_optimizer = torch.optim.Adam(self.critic_target.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.scale = reward_scale

        self.total_it = 0
        self.exploration = 0
        self.exploration_total = 2000

    def reset_memory(self):
        self.exploration = 0
        self.exploration_total = 2000
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

    def store_transition(self, curr_state, action, next_state, reward, done):
        self.replay_buffer.add(curr_state, action, next_state, reward, done)

    def select_action(self, state):
        if self.exploration < self.exploration_total:
            self.exploration += 1
            return np.random.randint(0, 2, 200)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1.view(-1), target_Q2.view(-1)).view(-1)
            target_Q = self.scale * reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1.view(-1), target_Q) + F.mse_loss(current_Q2.view(-1), target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        self.critic_target_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_target_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            self.actor_target_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()
            self.actor_target_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, filename='./save_model/TD3/'):
        torch.save(self.critic.state_dict(), filename + "_critic_" + str(self.user_id))
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer_" + str(self.user_id))

        torch.save(self.actor.state_dict(), filename + "_actor_" + str(self.user_id))
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer_" + str(self.user_id))

    def load_models(self, filename='./save_model/TD3/'):
        self.critic.load_state_dict(torch.load(filename + "_critic_" + str(self.user_id)))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer_" + str(self.user_id)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor_" + str(self.user_id)))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer_" + str(self.user_id)))
        self.actor_target = copy.deepcopy(self.actor)
