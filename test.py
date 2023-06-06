# Barbara
# 开发时间：2023/5/29 18:45

import os
import argparse
import time
import gym
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer
from tjdqn import DQN
from tensorboardX import SummaryWriter

# def change_to_tensor(data_np, dtype=torch.float32):
#     """
#     change numpy array to torch.tensor
#     :param dtype:
#     :param data_np:
#     :return:
#     """
#     data_tensor = torch.from_numpy(data_np).type(dtype)
#     if torch.cuda.is_available():
#         data_tensor = data_tensor.cuda()
#     return data_tensor

model_path = r"D:\nankai\DQN\tj\dqn_model.pth"

if __name__ == "__main__":

    writer = SummaryWriter("logs")

    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.RecordEpisodeStatistics(env)  # 记录每个episode的信息
    env = gym.wrappers.ResizeObservation(env, (84, 84))  # 将图像大小调整为84*84
    env = gym.wrappers.GrayScaleObservation(env)  # 将图像转换为灰度图
    env = gym.wrappers.FrameStack(env, 4)  # 将连续的4帧图像叠加起来
    env = MaxAndSkipEnv(env, skip=4)  # 每4帧图像执行一次动作

    rb = ReplayBuffer(1_000_00, env.observation_space, env.action_space, 'cuda',
                      optimize_memory_usage=True, handle_timeout_termination=False)


    # initial model
    q_network = DQN(env.action_space.n).to("cuda")
    q_network.load_state_dict(torch.load(model_path))

    total_steps = 0
    total_rewards = 0  # 总奖励
    episode = 1

    dead = False  # 是否死亡


    # 初始化状态s1 = {x1}和预处理后的状态φ1 = φ(s1)
    obs = env.reset()

    # 随机执行一定数量的Noop和fire操作，以便于重置环境
    for _ in range(random.randint(1, 30)):
        obs, _, _, info = env.step(1)

    while episode < 1000:

        dead = False        # 是否死亡
        total_rewards = 0   # 总奖励

        # 初始化状态s1 = {x1}和预处理后的状态φ1 = φ(s1)
        obs = env.reset()

        # 随机执行一定数量的Noop和fire操作，以便于重置环境
        for _ in range(random.randint(1, 30)):
            obs, _, _, info = env.step(1)

        while not dead:

            # 当前生命值/生命数
            current_life = info['lives']

            # 有ε的概率随机选择动作a，否则选择a = max_a Q∗(φ(st), a; θ)
            if random.random() < 0.05:
                action = np.array(env.action_space.sample())
            else:
                q_values = q_network(torch.Tensor(obs).unsqueeze(0).to('cuda'))
                action = torch.argmax(q_values, dim=1).item()

            # 在模拟器中执行动作a，并观察奖励r_t和图像x_{t+1}
            next_obs, reward, dead, info = env.step(action)

            # 如果生命值减少，则设置终止标志位为True
            done = True if (info['lives'] < current_life) else False

            # 预处理后的状态φ_{t+1} = φ(s_{t+1})
            real_next_obs = next_obs.copy()

            total_rewards += reward
            reward = np.sign(reward)  # 把奖励限制在[-1, 1]之间
            total_steps += 1

            # 把转移(φt, at, rt, φt+1)存储在D中
            rb.add(obs, real_next_obs, action, reward, done, info)

            obs = next_obs

        prewards = total_rewards.it

        print("Episode %d done in %d steps, total reward %.2f" % (episode, total_steps, total_rewards))
        time.sleep(1)
        env.reset()
        if episode > 200:
            break
        writer.add_scalars("test_result", {'rewards': total_rewards, 'steps': total_steps}, episode)
        writer.add_text('X_AXIS', 'episode')
        writer.add_text('Y_AXIS', 'rewards & steps')
        # writer.add_scalar("total_steps", total_steps, episode)
        episode += 1
        total_rewards = 0
        total_steps = 0

