import numpy as np
import csv
from torch.backends import cudnn
import logging
import collections

from nn_model.replay_memory import ReplayBuffer
from nn_model.DDPG import DDPG
from nn_model.our_DDPG import DDPG_V2
from nn_model.TD3 import TD3
from nn_model.DQN import DQN
from nn_model.PPO import PPO
from nn_model.AC import AC
from nn_model.DDQN import DDQN
from nn_model.SAC import SAC
from nn_model.RVI import RVI

cudnn.benchmark = True


class Agent:

    def __init__(self, config, env):
        self.config = config
        self.lambda_U = self.config.lambda_U
        self.phi_U = self.config.phi_U
        self.capacity = self.config.capacity
        self.user_id = self.config.userId
        self.redundancy = self.config.redundancy
        self.logger = logging.getLogger("Agent for SOTA solution")

        # define environment
        self.env = env

        kwargs = {
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "max_action": self.config.max_action,
            "user_id": self.config.userId,
            "discount": self.config.discount,
            "batch_size": self.config.batch_size,
        }

        # Initialize policy
        if self.config.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = self.config.policy_noise * self.config.max_action
            kwargs["noise_clip"] = self.config.noise_clip * self.config.max_action
            kwargs["policy_freq"] = self.config.policy_freq
            self.policy_model = TD3(**kwargs)
        elif self.config.policy == "OurDDPG":
            self.policy_model = DDPG_V2(**kwargs)
        elif self.config.policy == "DDPG":
            self.policy_model = DDPG(**kwargs)
        elif self.config.policy == "DQN":
            kwargs.pop("max_action")
            kwargs["learn_rate"] = self.config.lrRL
            self.policy_model = DQN(**kwargs)
        elif self.config.policy == "PPO":
            kwargs.pop("max_action")
            kwargs["n_epochs"] = self.config.epoch_rl
            self.policy_model = PPO(**kwargs)
        elif self.config.policy == "SAC":
            kwargs["beta"] = self.config.lrRL
            self.policy_model = SAC(**kwargs)
        elif self.config.policy == "AC":
            self.policy_model = AC(**kwargs)
        elif self.config.policy == "DDQN":
            kwargs.pop("max_action")
            kwargs["alpha"] = self.config.alpha
            kwargs["epsilon"] = self.config.epsilon
            kwargs["mem_size"] = self.config.mem_size
            self.policy_model = DDQN(**kwargs)
        elif self.config.policy == "RVI":
            kwargs["beta"] = self.config.lrRL
            # kwargs.pop("max_action")
            self.policy_model = RVI(**kwargs)

        # state, done = self.env.reset(), False
        self.threshold = 0
        self.out_reward = 200
        self.print_iteration = 1000
        self.episode_num = self.config.max_epoch_rl

        # define memory
        self.memory = ReplayBuffer(self.config.state_dim, self.config.action_dim)

        self.batch_size = self.config.batch_size

        self.cuda = self.config.cuda
        self.reward = 0
        self.record_time = 1

        log_filename = self.config.log_path + self.config.policy + '_' \
                       + str(self.config.videoId) + '_performance_' + str(self.user_id) + '.csv'
        f1 = open(log_filename, 'w', encoding='utf-8')
        self.csv_writer = csv.writer(f1)
        self.csv_writer.writerow(["accuracy", "precision", "recall", "predicted tile"])

        reward_filename = self.config.log_path + self.config.policy + '_reward_' \
                          + str(self.config.videoId) + '_' + str(self.user_id) + '.csv'
        f2 = open(reward_filename, 'w', encoding='utf-8')
        self.csv_writer_reward = csv.writer(f2)
        self.csv_writer_reward.writerow(["reward"])

    '''
    def load_checkpoint(self, index, file_name="checkpoint"):
        filename = self.config.model_path + file_name + str(index) + '.pth.tar'
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.episode_num = checkpoint['episode']
            self.policy_model.load_state_dict(checkpoint['state_dict'])
            # self.optim.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.model_path, checkpoint['episode'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.model_path))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint", is_best=0):
        state = {
            'episode': self.episode_num,
            'state_dict': self.policy_model.state_dict(),
            # 'optimizer': self.optim.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.model_path + file_name + str(self.user_id) + '.pth.tar')
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.model_path + file_name,
                            self.config.model_path + 'model_best.pth.tar')
    '''

    def federate_learning(self, models):
        fl_model = models[0]
        worker_state_dict = [x.state_dict() for x in models]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(models)):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / len(models)
        # update fed weights to fl model
        fl_model.load_state_dict(fed_state_dict)
        return fl_model

    def load_checkpoint(self, policy_model):
        print('.... federate learning ....')
        if self.config.policy == "SAC":
            model_remote_actor = [policy_model.actor, self.policy_model.actor]
            model_remote_critic_1 = [policy_model.critic_1, self.policy_model.critic_1]
            model_remote_critic_2 = [policy_model.critic_2, self.policy_model.critic_2]
            model_remote_value = [policy_model.value, self.policy_model.value]
            model_remote_target_value = [policy_model.target_value, self.policy_model.target_value]

            self.policy_model = policy_model
            self.policy_model.actor = self.federate_learning(model_remote_actor)
            self.policy_model.critic_1 = self.federate_learning(model_remote_critic_1)
            self.policy_model.critic_2 = self.federate_learning(model_remote_critic_2)
            self.policy_model.value = self.federate_learning(model_remote_value)
            self.policy_model.target_value = self.federate_learning(model_remote_target_value)
        elif self.config.policy == "TD3":
            model_remote_actor = [policy_model.actor, self.policy_model.actor]
            model_remote_actor_target = [policy_model.actor_target, self.policy_model.actor_target]
            model_remote_critic = [policy_model.critic, self.policy_model.critic]
            model_remote_critic_target = [policy_model.critic_target, self.policy_model.critic_target]

            self.policy_model = policy_model
            self.policy_model.actor = self.federate_learning(model_remote_actor)
            self.policy_model.actor_target = self.federate_learning(model_remote_actor_target)
            self.policy_model.critic = self.federate_learning(model_remote_critic)
            self.policy_model.critic_target = self.federate_learning(model_remote_critic_target)
        else:
            self.policy_model = policy_model
        self.policy_model.reset_memory()

    def save_checkpoint(self, is_best=0):
        self.policy_model.save_models()

    def out_print(self, TP, TN, FP, FN, write):
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        if (TP + FP) != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        print("node ID:", str(self.user_id),
              "accuracy:", str(accuracy),
              "precision:", str(precision),
              "recall", str(recall),
              "predicted tile", str(TP + FP))

        if write:
            self.csv_writer.writerow([str(accuracy), str(precision), str(recall), str(TP + FP)])

    def out_print_reward(self, reward, write):

        if write:
            self.csv_writer_reward.writerow([str(reward)])

    def train_one_epoch(self, curr_state, next_state):

        episode_timesteps = 0
        episode_reward = 0
        action_out = []
        best_reward = -1e5
        best_action = np.random.randint(0, 2, 200)

        while True:
            TP, TN, FP, FN = 0, 0, 0, 0
            episode_timesteps += 1

            if self.config.policy == 'PPO':
                action, prob, val = self.policy_model.select_action(np.array(curr_state))
            else:
                action = (self.policy_model.select_action(np.array(curr_state))).clip(-1, 1)

            penalty = 0
            reward = 0
            for index, value in enumerate(action):
                if value > self.threshold:
                    tmp = 1
                else:
                    tmp = 0

                if tmp == next_state[index]:
                    if tmp == 1:
                        TP += 1
                        reward += 2
                        penalty += -2
                    else:
                        TN += 1
                        reward += 1
                        penalty += -1
                else:
                    if tmp == 1:
                        FP += 1
                        reward += -20
                        penalty += 30
                    else:
                        FN += 1
                        reward += -10
                        penalty += 15

            reward = (TP * TP - max(self.phi_U * ((TP + FP) - self.capacity), 0)) - self.lambda_U * penalty

            '''
            if torch.is_tensor(action) and self.cuda:
                action_np = action.cpu().detach().numpy()
            elif type(action) is np.ndarray:
                action_np = action
            else:
                action_np = action.detach().numpy()
                
            total_num_tiles = int(sum(next_state) * self.redundancy)
            actu_index_list = [k for k, x in enumerate(next_state) if x == 1]
            pred_index_list = np.argsort(action_np)[0:total_num_tiles].tolist()
            right_pred = len(set(actu_index_list) & set(pred_index_list))
            wrong_pred = total_num_tiles - right_pred
            
            reward = right_pred - 10 * wrong_pred
            '''
            """
            if reward > 0:
                self.record_time += 1
                self.reward = (self.reward + reward) / self.record_time
                reward = reward - self.reward
            """

            if best_reward < reward:
                best_reward = reward
                best_action = action

            # Store data in replay buffer
            if self.config.policy == 'PPO':
                self.policy_model.store_transition(curr_state, action, prob, val, reward, False)
            else:
                self.policy_model.store_transition(curr_state, action, next_state, reward, False)

            if episode_timesteps > self.batch_size:
                self.policy_model.train()

            episode_reward += reward

            self.out_print_reward(reward, True)

            if episode_timesteps % self.print_iteration == 0 and self.print_iteration != 0:
                print(f"Episode T: {episode_timesteps} "
                      f"Slot Reward: {best_reward:.1f} "
                      f"Total Reward: {episode_reward:.1f}")
                self.out_print(TP, TN, FP, FN, False)

            if max(self.episode_num, self.print_iteration) + 1 < episode_timesteps or best_reward > self.out_reward:
                for index, value in enumerate(best_action):
                    if value > self.threshold:
                        action_out.append(1)
                    else:
                        action_out.append(0)
                self.out_print(TP, TN, FP, FN, True)
                self.policy_model.reset_memory()
                break

        return action_out, reward