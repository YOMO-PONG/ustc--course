import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections

# hyper-parameters
EPISODES = 2000                 # 训练/测试幕数
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_IETRATION = 1000         # 保存Checkpoint的间隔
MEMORY_CAPACITY = 10000         # Memory的容量
MIN_CAPACITY = 500              # 开始学习的最小记忆容量。这意味着在记忆库中积累了500个经验之后，才开始学习过程
Q_NETWORK_ITERATION = 10        # 同步target network的间隔 同步目标网络（target network）的间隔。每10次学习迭代后，目标网络的参数将被更新为评估网络（evaluation network）的参数。
EPSILON = 0.01                  # epsilon-greedy
SEED = 0
MODEL_PATH = './Model/'
SAVE_PATH_PREFIX = './log/dqn/'
TEST = False # 测试标志。如果设置为True，模型将在测试模式下运行，不会进行学习过程。


env = gym.make('CartPole-v1', render_mode="human" if TEST else None)
# env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)


random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

NUM_ACTIONS = env.action_space.n  # 2
NUM_STATES = env.observation_space.shape[0]  # 4
ENV_A_SHAPE = 0 if np.issubdtype(type(env.action_space.sample()), np.integer) else env.action_space.sample().shape  # 0, to confirm the shape

# 定义网络模型，用于近似Q函数
class Model(nn.Module):
    def __init__(self, num_inputs=4):
        super(Model, self).__init__()
        self.linear = nn.Linear(NUM_STATES, 512)
        self.linear2 = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class Data:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __iter__(self):
        # 返回对象的各个部分的迭代器
        for attr in [self.state, self.action, self.reward, self.next_state, self.done]:
            yield attr

# 实现一个循环队列，用于存储和抽样经验，
class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        # TODO
        self.buffer.append(data)
        pass
    
    def get(self, batch_size):
        # TODO
        batch = random.sample(self.buffer,batch_size)
        return batch
        pass
        


class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Model().to(device), Model().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPSILON = 1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        else: 
            # random policy
            action = np.random.randint(0,NUM_ACTIONS)  # int random number
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

#存储经验至经验库中
    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

# 从经验库中抽取小批量经验用于学习，并定期同步目标网络的参数
    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            print(f"learn_step.counter:{self.learn_step_counter}")
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # TODO
        batch = self.memory.get(BATCH_SIZE)
        """
        for state, action, reward, next_state, done in batch:
            action_value = self.eval_net.forward(state)
            value_pred = action_value[action]
            action_value_target = self.target_net.forward(next_state)
            value_target = torch.max(action_value_target)+reward
            loss += loss_func(value_pred,value_target)
        loss = loss/BATCH_SIZE
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将数据转换为适当的Tensor格式
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(device)  # 1 for done, 0 otherwise

        # 计算当前状态的Q值
        current_q_values = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # 计算下一个状态的最大预期Q值
        # 使用在线网络选择下一个状态的最佳动作
        next_actions = self.eval_net(next_states).max(1)[1]
        # 使用目标网络来评估这个被选择动作的 Q 值
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        next_q_values[dones] = 0.0  # 对于结束的状态，其价值为0
        expected_q_values = rewards + GAMMA * next_q_values

        # 计算损失
        loss = self.loss_func(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))

def main():
    dqn = DQN()
    
    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')

    if TEST:
        dqn.load_net(MODEL_PATH)
    for i in range(EPISODES):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED)

        ep_reward = 0
        while True:
            action = dqn.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)  # choose best action
            next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            if TEST:
                env.render()
            if dqn.memory_counter >= MIN_CAPACITY and not TEST:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                if TEST:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break
            state = next_state
        writer.add_scalar('reward', ep_reward, global_step=i)


if __name__ == '__main__':
    main()