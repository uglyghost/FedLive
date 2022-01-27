import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_model.replay_memory import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


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

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPG_V2(object):
	def __init__(self,
				 state_dim,
				 action_dim,
				 max_action,
				 user_id,
				 batch_size=64,
				 discount=0.99,
				 tau=0.005,
				 reward_scale=2
				 ):

		self.user_id = user_id
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
		self.batch_size = batch_size
		self.replay_buffer = ReplayBuffer(state_dim, action_dim)

		self.discount = discount
		self.tau = tau
		self.exploration = 0
		self.exploration_total = 1000
		self.scale = reward_scale

	def reset_memory(self):
		self.exploration = 0
		self.exploration_total = 1000
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
		# Sample replay buffer 
		state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = self.scale * reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save_models(self, filename='./save_model/DDPG/'):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load_models(self, filename='./save_model/DDPG/'):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		