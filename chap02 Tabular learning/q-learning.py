#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: q-learning.py
@time: 7/30/20 2:53 PM
@desc:
'''

import gym
import collections
from tensorboardX import SummaryWriter
import argparse

class Agent:
    def __init__(self, args):
        self.env = gym.make(args.env)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):

        action_values = [self.values[(state, action)] for action in range(self.env.action_space.n)]

        best_value = max(action_values)
        best_action = action_values.index(best_value)

        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + args.gamma * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-args.alpha) + new_val * args.alpha

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = next_state
        return total_reward

def main(args):
    test_env = gym.make(args.env)
    agent = Agent(args=args)
    writer = SummaryWriter(comment="-q-learning")

    iter = 0
    best_reward = 0.0
    while True:
        iter += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(args.Test_Episodes):
            reward += agent.play_episode(test_env)
        reward /= args.Test_Episodes
        writer.add_scalar("reward", reward, iter)
        if reward > best_reward:
            print("Best reward updated {} -> {}".format(best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in {} iterations!".format(iter))
            break
    writer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="The Parameter of Value Iteration")

    parser.add_argument("--env", type=str, help="the name of environment", default="FrozenLake-v0")
    parser.add_argument("--gamma", type=float, help="The discount factor", default=0.9)
    parser.add_argument("--Test_Episodes", type=int, help="Test_Episodes decided whether stop training", default=20)
    parser.add_argument("--alpha", type=float, help="the learning efficiency", default=0.2)

    args = parser.parse_args()
    main(args)