#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: dqn_pong.py
@time: 7/31/20 2:28 PM
@desc:
'''

from lib import wrappers, dqn_model

import argparse
import time
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

Experience = collections.namedtuple(typename='Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])

class ExperienceBuffer:
    def __init__(self, args):
        self.buffer = collections.deque(maxlen=args.replay_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        randomly sample the batch of transitions from the replay buffer.
        :param batch_size:
        :return:
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               dones, np.array(next_states)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_episode_step(self, net, epsilon=0.0, device=None):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device=None,args=None):
    states, actions, rewards, dones, next_states = batch

    # preceding data
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)


    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]

    # last step in the episode doesn't have a discounted reward of the next state
    next_state_values[done_mask] = 0.0

    # detach() function to prevent gradients from flowing into the target network's graph
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * args.gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="The parameter of DQN")
    parser.add_argument("--env", type=str, help="The default env name", default="PongNoFrameskip-v4")
    parser.add_argument("--MRB", type=float, help="Mean reward boundary for stop of training", default=19.0)
    parser.add_argument("--gamma", type=float, help="gamma value used for Bellman approximation", default=0.99)
    parser.add_argument("--batch_size", type=int, help="batch size sampled from the replay buffer", default=32)
    parser.add_argument("--replay_size", type=int, help="maximum capacity of the buffer", default=10000)
    parser.add_argument("--repaly_start_size", type=int, help="The count of frames we wait for before starting training", default=10000)
    parser.add_argument("--lr", type=float, help="learning rate used in the Adam optimizer", default=1e-4)
    parser.add_argument("--sync_target", type=int, help="frequently we sync model weights from training to target", default=1000)

    parser.add_argument("--epsilon_decay", type=int, help="the first 150,000 frames, epsilon is linearly decayed", default=150000)
    parser.add_argument("--epsilon_start", type=float, help="start value with epsilon", default=1.0)
    parser.add_argument("--epsilon_end", type=float, help="end value with epsilon", default=0.01)
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env)

    buffer = ExperienceBuffer(args=args)

    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(args.epsilon_end, args.epsilon_start - frame_idx / args.epsilon_decay)

        reward = agent.play_episode_step(net, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)

            # Speed as a count of frames processed per second
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()

            # Mean reward for the last 100 episodes
            mean_reward = np.mean(total_rewards[-100:])
            print("{}: done {} games, mean reward {}, eps {}, speed {} f/s".format(frame_idx, len(total_rewards), mean_reward, epsilon, speed))

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward

            # the boundary is 19.0, which means winning more than 19 from 21 possible games
            if mean_reward > args.MRB:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < args.replay_size:
            continue

        if frame_idx % args.sync_target == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device=device, args=args)
        loss_t.backward()
        optimizer.step()

    writer.close()


