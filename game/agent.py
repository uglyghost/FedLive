import math
import random
import shutil

import numpy as np
import torch
from torch.backends import cudnn

import logging
from torch.autograd import Variable

from game.huber_loss import HuberLoss
from game.replay_memory import ReplayMemory, Transition

cudnn.benchmark = True


class DQNAgent:

    def __init__(self, config, env):
        self.config = config

        self.logger = logging.getLogger("DQNAgent")

        # define memory
        self.memory = ReplayMemory(self.config)

        # define loss
        self.loss = HuberLoss()

        # define optimizer
        self.optim = torch.optim.RMSprop(self.policy_model.parameters())

        # define environment
        self.env = env

        # initialize counter
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        self.batch_size = self.config.batch_size

        self.cuda = self.config.cuda
        self.reward = 0
        self.record_time = 1

        if self.cuda:
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
        else:
            self.logger.info("Program will run on *****CPU***** ")
            self.device = torch.device("cpu")

        self.policy_model = self.policy_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Initialize Target model with policy model state dict
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

    def load_checkpoint(self, file_name="checkpoint.pth.tar"):
        filename = self.config.model_path + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_episode = checkpoint['episode']
            self.current_iteration = checkpoint['iteration']
            self.policy_model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.model_path, checkpoint['episode'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.model_path))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'episode': self.current_episode,
            'iteration': self.current_iteration,
            'state_dict': self.policy_model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.model_path + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.model_path + file_name,
                            self.config.model_path + 'model_best.pth.tar')

    def select_action(self, state):
        """
        The action selection function, it either uses the model to choose an action or samples one uniformly.
        :param state: current state of the model
        :return:
        """

        # state = state.reshape(2, 8, 5, 5)
        sample = random.random()
        eps_threshold = self.config.eps_start + (self.config.eps_start - self.config.eps_end) * math.exp(
            -1. * self.current_iteration / self.config.eps_decay)
        self.current_iteration += 1
        if sample > eps_threshold:
            with torch.no_grad():
                _, _, _, conv_output_list_ret = self.policy_model(state)
                return conv_output_list_ret
                # return self.policy_model(state)[3].reshape(200)  # size (1,1)
        else:
            # _, _, _, conv_output_list_ret = self.policy_model(state)
            # return conv_output_list_ret.reshape(200)
            return torch.tensor(np.random.randint(0, 2, [1, 8, 1, 5, 5]), device=self.device, dtype=torch.long)

    def optimize_policy_model(self):
        """
        performs a single step of optimization for the policy model
        :return:
        """
        if self.memory.length() < self.batch_size:
            return
        # sample a batch
        transitions = self.memory.sample_batch(self.batch_size)

        one_batch = Transition(*zip(*transitions))

        # create a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, one_batch.next_state)),
                                      device=self.device,
                                      dtype=torch.uint8)  # [128]
        non_final_next_states = torch.cat([s for s in one_batch.next_state if s is not None])  # [< 128, 3, 40, 80]

        # concatenate all batch elements into one
        state_batch = torch.cat(one_batch.state)  # [128, 3, 40, 80]
        # action_batch = torch.cat(one_batch.action)  # [128, 1]
        reward_batch = torch.cat(one_batch.reward)  # [128]
        state_batch = state_batch.to(self.device)

        non_final_next_states = non_final_next_states.to(self.device)
        non_final_next_states_tmp = non_final_next_states[:, :, 1, :, :]
        non_final_next_states_tmp = non_final_next_states_tmp.reshape([256, 200])

        curr_state_action_values = reward_batch  # [128, 1]

        # Get V(s_{t+1}) for all next states. By definition we set V(s)=0 if s is a terminal state.
        # next_state_values = torch.zeros(self.batch_size, 1, 5, 5, device=self.device)  # [128]
        _, _, _, non_final_next_states_value = self.target_model(state_batch)
        non_final_next_states_value = non_final_next_states_value.reshape([256, 200])

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        for i, v1 in enumerate(non_final_next_states_value):
            next_state_tmp = non_final_next_states_tmp[i, :]
            for j, v2 in enumerate(v1):
                if v2 == next_state_tmp[j]:
                    next_state_values[i] += 1
                else:
                    next_state_values[i] += -1

        # Get the expected Q values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch  # [128]
        # compute loss: temporal difference error
        curr_state_action_values = Variable(curr_state_action_values, requires_grad=True)
        expected_state_action_values = Variable(expected_state_action_values, requires_grad=True)
        loss = self.loss(curr_state_action_values, expected_state_action_values)

        # optimizer step
        self.optim.zero_grad()
        loss.backward()
        '''
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optim.step()
        '''
        return loss

    def list2tensor(self, state):
        target = np.array(state)  # list to numpy.array
        target = target.reshape(1, 2, 8, 5, 5)
        target = torch.from_numpy(target)  # array2tensor
        target = target.transpose(1, 2)
        if self.cuda:
            state = target.cuda()
            state = state.type(torch.cuda.FloatTensor)

        return state

    def train_one_epoch(self, curr_state, next_state):
        """
        One episode of training; it samples an action, observe next screen and optimize the model once
        :return:
        """
        current_episode = 0
        episode_duration = self.config.num_episodes
        curr_state_input = self.list2tensor(curr_state)
        next_state_input = self.list2tensor(next_state)

        while True:
            current_episode += 1
            # select action
            action = self.select_action(curr_state_input)
            # perform action and get reward

            action_test = action.reshape(200)

            reward = 0
            for index, value in enumerate(action_test):
                if value == next_state[index]:
                    reward += 1
                else:
                    reward += -1

            if reward > 0:
                self.record_time += 1
                self.reward = (self.reward + reward) / self.record_time
                reward = reward - self.reward
            else:
                reward = 0

            self.memory.push_transition(curr_state_input,
                                        action,
                                        next_state_input,
                                        torch.Tensor([reward]).cuda())

            # Policy model optimization step
            curr_loss = self.optimize_policy_model()
            if curr_loss is not None:
                if self.cuda:
                    curr_loss = curr_loss.cpu()
                print("Temporal Difference Loss", curr_loss.detach().numpy(), current_episode)
            else:
                print("Training Episode Duration", episode_duration, current_episode)

            if current_episode == episode_duration:
                break

        return action_test, reward
