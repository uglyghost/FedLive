from argparse import ArgumentParser

parser = ArgumentParser(description='Online viewport prediction with RL')

# basic arguments
parser.add_argument('--cuda', default='GPU', type=bool, help='whether cuda is in use')
parser.add_argument('--groupId', default=1, type=int, help='select experiment group')
parser.add_argument('--videoId', default=1, type=int, help='select video')
parser.add_argument('--policy', default='RVI', type=str, help='select method')
parser.add_argument('--redundancy', default=1.5, type=int, help='redundancy rate for selected tiles')

# utility function
parser.add_argument('--lambda_U', default=1, type=float, help='weight factor for QoE utility')
parser.add_argument('--phi_U', default=10, type=int, help='weight factor for bandwidth penalty')
parser.add_argument('--capacity', default=40, type=int, help='link capacity')

# basic arguments for CNN
parser.add_argument('--epochCNN', default=10, type=float, help='the epoch for CNN')
parser.add_argument('--lrCNN', default=0.001, type=float, help='learning rate for CNN model')
parser.add_argument('--sampleRate', default=1, type=int, help='how many frame of each')
parser.add_argument('--saliTrainNum', default=5, type=int, help='number of user for CNN model training')
parser.add_argument('--saliTestNum', default=10, type=int, help='number of user for CNN model test')
parser.add_argument('--loadPerModel', default=False, type=bool, help='whether load the previous model')

# basic arguments for RL
parser.add_argument('--epoch_rl', default=10, type=float, help='the epoch for RL model')
parser.add_argument('--max_epoch_rl', default=20000, type=float, help='the max epoch for RL train')
parser.add_argument('--lrRL', default=0.001, type=float, help='learning rate for RL model')
parser.add_argument('--rlUsrNum', default=8, type=int, help='select user for online RL model training and test')
parser.add_argument('--visId', default=8, type=int, help='visualization user ID')
parser.add_argument('--state_dim', default=400, type=int, help='State dimension')
parser.add_argument('--action_dim', default=200, type=int, help='Action dimension')
parser.add_argument('--max_action', default=1, type=int, help='Max value of action number')
parser.add_argument('--discount', default=1.0, type=float, help='Discount factor')
parser.add_argument('--tau', default=0.005, type=float, help='Target network update rate')

# TD3
parser.add_argument('--policy_noise', default=0.2, type=int, help='Noise added to target policy during critic update')
parser.add_argument('--noise_clip', default=0.5, type=int, help='Range to clip target policy noise')
parser.add_argument('--policy_freq', default=2, type=int, help='Frequency of delayed policy updates')

# AC
parser.add_argument('--alpha', default=0.00001, type=float, help='alpha for AC')
parser.add_argument('--beta', default=0.00001, type=float, help='beta for AC')

# DQN
parser.add_argument('--seed', default=1337, type=int, help='')
parser.add_argument('--eps_start', default=0.9, type=float, help='')
parser.add_argument('--eps_end', default=0.05, type=float, help='')
parser.add_argument('--eps_decay', default=200, type=int, help='')
parser.add_argument('--batch_size', default=256, type=int, help='')
parser.add_argument('--target_update', default=10, type=int, help='')
parser.add_argument('--num_episodes', default=100, type=int, help='')
parser.add_argument('--max_epoch', default=100, type=int, help='')
parser.add_argument('--gpu_device', default=0, type=int, help='')
parser.add_argument('--input_channels', default=8, type=int, help='')
parser.add_argument('--conv_filters', default=[8, 16, 32], type=list, help='')
parser.add_argument('--num_classes', default=2, type=int, help='')
parser.add_argument('--memory_capacity', default=1000, type=int, help='')
parser.add_argument('--save_iteration', default=20, type=int, help='')
parser.add_argument('--load_rl_model', default=False, type=bool, help='')

# DDQN
parser.add_argument('--epsilon', default=1.0, type=float, help='')
parser.add_argument('--mem_size', default=1000, type=int, help='')

# convLSTM
parser.add_argument('--VGG_indim', default=3, type=int, help='input size size for VGG')
parser.add_argument('--VGG_outdim', default=1, type=int, help='input size size for VGG')
parser.add_argument('--input_size', default=2, type=int, help='input size size for ConvLSTM')
parser.add_argument('--hidden_size', default=6, type=int, help='hidden size for ConvLSTM')
parser.add_argument('--numLayers', default=1, type=int, help='layers for ConvLSTM')

# cluster
parser.add_argument('--threshold', default=400, type=int, help='threshold of viewing similarity')

# save model iteration
parser.add_argument('--saveIter', default=100, type=int, help='the steps of saving model')

# dataset path settings
parser.add_argument('--saliency_path', default='D:/Multimedia/FoV_Prediction/Dataset/Saliency/',
                    type=str, metavar='PATH',
                    help='path to saliency feature records')
parser.add_argument('--records_path', default='D:/Multimedia/FoV_Prediction/Dataset/VRdataset/Experiment_',
                    type=str, metavar='PATH',
                    help='path to users viewport records')
parser.add_argument('--videos_path', default='D:/Multimedia/FoV_Prediction/Dataset/Videos/',
                    type=str, metavar='PATH',
                    help='path to video')
parser.add_argument('--frames_path', default='D:/Multimedia/FoV_Prediction/Dataset/frames/',
                    type=str, metavar='PATH',
                    help='path to video')

# other path settings
parser.add_argument('--model_path', default='./save_model/',
                    type=str, metavar='PATH',
                    help='path to trained model')
parser.add_argument('--checkpoint_file', default='./save_model/checkpoint.pth.tar',
                    type=str, metavar='PATH',
                    help='')
parser.add_argument('--log_path', default='./log/',
                    type=str, metavar='PATH',
                    help='path to save log csv')

args = parser.parse_args()


def get_args():
    arguments = parser.parse_args()
    return arguments
