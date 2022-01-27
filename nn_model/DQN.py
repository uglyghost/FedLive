import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir='./save_model/DQN'):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dqn')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.max_action * T.tanh(self.fc3(x))

        return actions

    def Q1(self, state, action):
        sa = T.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class DQN:
    def __init__(self, state_dim, action_dim, discount, learn_rate, batch_size,
                 epsilon=1.0, max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = discount
        self.epsilon = epsilon
        self.state_dim = [state_dim]
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = learn_rate
        self.action_space = [i for i in range(action_dim)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(learn_rate, n_actions=action_dim, input_dims=self.state_dim,
                                   fc1_dims=256, fc2_dims=256, name='q_eval',)
        self.Q_next = DeepQNetwork(learn_rate, n_actions=action_dim, input_dims=self.state_dim,
                                   fc1_dims=64, fc2_dims=64, name='q_next',)

        self.action_dim = action_dim

        self.state_memory = np.zeros((self.mem_size, *self.state_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.state_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *[action_dim]), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        self.exploration = 0
        self.exploration_total = 1000

    def reset_memory(self):
        self.exploration = 0
        self.exploration_total = 1000

        self.state_memory = np.zeros((self.mem_size, *self.state_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.state_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *[self.action_dim]), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, curr_state, action, next_state, reward, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = curr_state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def select_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state.float())
            # action = T.argmax(actions).item()
            action = actions.cpu().detach().numpy().reshape([200])
        else:
            # action = np.random.choice(self.action_space)
            action = np.random.randint(0, 2, [200])

        return action

    def train(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_eval_actions = self.Q_eval.forward(state_batch)
        q_eval = sum((q_eval_actions - new_state_batch[:, 0:200]).t())

        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()