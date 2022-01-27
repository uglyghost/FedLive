import math
import numpy as np
import gym

from arguments import get_args
from live_CNN import LiveCNN
from utils.video_process import processed_data_200_Tiles_all

from game.agent_v2 import Agent


# from game.agent import Agent      # another solution such as convLSTM

# 单个用户的viewpoint获取
def viewerSali(solver, countMain=0):
    p_u = []
    if countMain + solver.bufLen < solver.totalFrames:
        for i in range(solver.bufLen):
            if i % solver.sampleRate == 0:
                p_u.append(solver.LocationPerFrame[math.ceil(solver.interFOV * (countMain - 1)) + 1])
            countMain += 1
        return p_u
    else:
        return []


# 整个用户集群的视野点构成显著性特征
def serverSali(solvers):
    all_u = []
    # 服务器端获取saliency
    for ii in range(0, math.floor(solvers[0].totalFrames / solvers[0].bufLen)):
        tmp_u = []
        average_u = np.zeros([25])
        for i, solver in enumerate(solvers):
            p_u = np.zeros([200])
            add_one = viewerSali(solver, ii)
            for j in range(25):
                au = processed_data_200_Tiles_all(solver.W_Frame,
                                                  solver.H_Frame,
                                                  solver.tileNO,
                                                  j + 1,
                                                  add_one)
                p_u[8 * j:8 * (j + 1)] = au

            if i == 0:
                average_u = p_u
            else:
                average_u = average_u + p_u

        for i, value in enumerate(average_u):
            if value > 0:
                tmp_u.append(1)
            else:
                tmp_u.append(0)

        all_u.append(tmp_u)

    return all_u


# 将viewpoint转化为200维向量
def stateToVec(solver, next_state):
    state_vec = np.zeros([200])
    add_one = next_state
    realSali = []
    for j in range(25):
        au = processed_data_200_Tiles_all(solver.W_Frame,
                                          solver.H_Frame,
                                          solver.tileNO,
                                          j + 1,
                                          add_one)
        if len(au) == 0:
            return realSali
        state_vec[8 * j:8 * (j + 1)] = au

    for i in range(8):
        for j in range(25):
            realSali.append(state_vec[j * 8 + i])  # 检查视野是否输出正确？行列是否正确？

    return realSali


def getState(viewer, saliency):
    # 获取观看者历史观看点
    view_point = env[j].getNextSate(viewer)

    # 获取用户的实际观看记录
    view_point_fix = []
    for index, value in enumerate(view_point):
        view_point_fix.append([value[0] / solvers[1].W_Frame, value[1] / solvers[1].H_Frame])

    # 历史观看和训练完的saliency作为状态
    next_vec = stateToVec(viewer, view_point)
    state_all = next_vec + saliency

    return state_all, view_point_fix


if __name__ == '__main__':

    args = get_args()

    solvers = []
    agents = []

    tileNum = args.sampleRate * 5 * 5

    # 生成所有用户的视频信息类
    for index in range(args.totalUser):
        args.userId = index + 1
        solvers.append(LiveCNN(args))

    for index in range(args.totalUser):
        # 为每个用户加载视频数据
        solvers[index].videoLoad()

    # 训练集
    trainSolvers = solvers[0:args.saliTrainNum - 1]
    train_sali_u = serverSali(trainSolvers)
    # 测试集
    testSolvers = solvers[args.saliTrainNum:args.saliTrainNum + args.saliTestNum - 1]
    test_sali_u = serverSali(testSolvers)

    resultSali = []

    '''
    # 画个图看看情况
    env = gym.make('MyEnv-v1')
    frames = []
    solvers[0].countMain = 0
    for i in range(int(solvers[0].totalFrames/8)):
        view_point = env.getNextSate(solvers[0])
        view_point_fix = []
        for index, value in enumerate(view_point):
            view_point_fix.append([value[0] / solvers[0].W_Frame, value[1] / solvers[0].H_Frame])

        next_vec = stateToVec(solvers[0], view_point)
        env.setPrediction(next_vec)
        env.setFov(view_point_fix)
        frames.append(env.render(mode='rgb_array'))

    # 保存到gif
    display_frames_as_gif(frames)

    env.close()
    '''

    # 加载CNN模型，用于预测saliency
    solvers[0].load_CNN_model()
    # solvers[1].load_CNN_model()
    result_8 = []
    for i, value in enumerate(train_sali_u):
        # 训练针对saliency的CNN网络
        if args.loadPerModel:
            solvers[0].load_sali_model()
        result = solvers[0].run_step(value)

        if i % args.saveIter == 0 and i != 0:
            print('step: ', i, " save model checkpoint")
            solvers[0].save_sali_model()
            # solvers[1].load_sali_model()
            # 测试saliency性能
            for j, value2 in enumerate(train_sali_u[i - args.saveIter:i]):
                solvers[0].run_test_step(value2)
        if i % 8 == 0 and i != 0:
            resultSali.append(result_8)
            result_8 = result
        else:
            result_8 = result_8 + result

    # 主要循环体
    env = []
    for index, viewers in enumerate(testSolvers):
        # 为每个观看者生成观看环境
        env.append(gym.make('MyEnv-v1'))
        args.userId = index + 1
        if args.load_rl_model:
            agents.append(Agent(args, env).load_checkpoint(file_name="checkpoint.pth.tar"))
        else:
            agents.append(Agent(args, env))

    reward = np.zeros(len(agents))
    rewardBest = np.zeros(len(agents))
    for i, saliency in enumerate(resultSali[1:-1]):
        for j, viewer in enumerate(testSolvers):

            curr_state, view_point_fix = getState(viewer, saliency)
            viewer_next = viewer
            next_state, _ = getState(viewer_next, saliency)
            env[j].setState(curr_state)

            # time-shift效应

            # 使用agent进行预测和在线训练
            prediction_vec, rewardTmp = agents[j].train_one_epoch(curr_state, next_state)

            reward[j] += rewardTmp

            if i % args.save_iteration == 0 & i != 0:
                print("Agent{0}, The model current reward: {1}, Best reward: {2}".format(j, reward[j], rewardBest[j]))
                if rewardBest <= reward[j]:
                    agents[i].save_checkpoin(is_best=1)
                    rewardBest[j] = reward[j]

            # The target network has its weights kept frozen most of the time
            if i % args.target_update == 0 & i != 0:
                agents[j].target_model.load_state_dict(agents[j].policy_model.state_dict())

            # 可视化某个用户的预测情况
            if args.visId == viewer.userId:
                env[j].setPrediction(prediction_vec)
                env[j].setFov(view_point_fix)
                env[j].render()

    # 关闭所有观看着环境
    for j in range(len(testSolvers)):
        env[j].close()
