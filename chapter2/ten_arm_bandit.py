#!/usr/bin/env python
"""Run a 10 arm bandit problem using action-value methods"""
########################################################################
# File: ten_arm_bandit.py
#  executable: ten_arm_bandit.py
#
# Author: Andrew Bailey
# History: 02/05/18 Created
########################################################################

import os
import numpy as np
import collections
import platform
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
if platform.system() == "Darwin":
    mpl.use("macosx")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Bandit(object):

    def __init__(self, n_bandits, reward_function, q_star=None):
        self.n_bandits = n_bandits
        self.q_t = [0] * self.n_bandits
        self.n_steps = [0] * self.n_bandits

        if q_star is not None:
            self.q_star = q_star
        else:
            self.q_star = np.random.normal(loc=0, scale=1, size=self.n_bandits)
        self.reward_function = reward_function
        assert self.reward_function == 0, "Only one reward function option right now"

    @staticmethod
    def calculate_new_q(q_t, step_size, reward):
        """Calculate the new action value given step size, previous action and new reward"""
        return q_t + (step_size * (reward-q_t))

    def select_action(self, epsilon):
        """Select an action. If epsilon is zero it will always take a greedy action"""
        take_random_action = np.random.binomial(1, epsilon)
        if take_random_action == 1:
            return np.random.random_integers(0, self.n_bandits-1)
        else:
            indices = [i for i, x in enumerate(self.q_t) if x == max(self.q_t)]
            if len(indices) == 1:
                return indices[0]
            else:
                random_choice = np.random.random_integers(0, len(indices)-1)
                return indices[random_choice]

    def get_reward(self, action_index):
        """Calculate reward for a given action"""
        if self.reward_function == 0:
            return self.reward_function_0(action_index)
        else:
            return 0

    def take_action(self, epsilon, step_size=None, add_noise=False):
        """Take an action, get reward, update n_steps and update value function"""
        action_index = self.select_action(epsilon)
        reward = self.get_reward(action_index)
        self.n_steps[action_index] += 1
        if step_size is None:
            step_size = 1 / self.n_steps[action_index]

        self.q_t[action_index] = self.calculate_new_q(self.q_t[action_index], step_size, reward)
        if add_noise:
            self.q_star = [self.q_star[i]+x for i, x in enumerate(np.random.normal(0, 0.01, self.n_bandits))]

        return action_index, reward

    def reward_function_0(self, action_index):
        """Return the reward function for function where the distributions were selected via a random sampling of
        the standard normal and variance is 1"""
        return np.random.normal(loc=self.q_star[action_index], scale=1)

    def reset(self, q_star=None):
        """Reset parameters for another iteration"""
        self.q_t = [0] * self.n_bandits
        self.n_steps = [0] * self.n_bandits
        if q_star:
            self.q_star = q_star


def plot_rewards(rewards, names, save_fig_dir=None, all_all_max_rewards=None):
    plt.figure(figsize=(20, 9))
    panel1 = plt.axes([0.1, 0.1, .8, .8])
    panel1.set_xlabel('Steps')
    panel1.set_ylabel('Average Reward')
    panel1.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    panel1.xaxis.set_major_locator(ticker.AutoLocator())
    panel1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for name, reward in zip(names, rewards):
        panel1.plot([np.mean(x) for x in zip(*reward)], label=name)
    if all_all_max_rewards is not None:
        for name, reward in zip(names, all_all_max_rewards):
            panel1.plot([np.mean(x) for x in zip(*reward)], label="Max_Reward - {}".format(name))

    panel1.legend(loc='upper right', fancybox=True, shadow=True)
    if save_fig_dir is not None:
        plt.savefig(os.path.join(save_fig_dir, "average_reward.png"))
    else:
        plt.show()


def plot_actions(rewards, names, save_fig_dir=None):
    plt.figure(figsize=(20, 9))
    panel1 = plt.axes([0.1, 0.1, .8, .8])
    panel1.set_xlabel('Steps')
    panel1.set_ylabel('Percent Correct Action')
    panel1.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    panel1.xaxis.set_major_locator(ticker.AutoLocator())
    panel1.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    for name, reward in zip(names, rewards):
        panel1.plot([np.mean(x) for x in zip(*reward)], label=name)
    panel1.legend(loc='upper right', fancybox=True, shadow=True)
    if save_fig_dir is not None:
        plt.savefig(os.path.join(save_fig_dir, "percent_correct_action.png"))
    else:
        plt.show()


def some_function(o, a):
    return o + a*(1 - o)


def main():
    step_size_alpha = 0.1
    n_bandits = 10
    save_fig_dir = "/Users/andrewbailey/CLionProjects/ReinforcementLearningHomework/chapter2"
    # save_fig_dir = None

    # epsilon is the probability of randomly selecting another action
    np.random.normal(loc=0, scale=0.1)
    q_star = np.random.normal(loc=0, scale=1, size=n_bandits)
    best_action2 = np.argmax(q_star)
    bh = Bandit(n_bandits, 0, q_star)
    i = 0
    j = 0
    all_all_rewards = []
    all_all_actions = []
    all_all_best_actions = []
    all_all_max_rewards = []
    all_all_picked_correct_action = []
    epsilons = [0.1, 0.01, 0]
    names = ["epsilon = 0.1", "epsilon = 0.01", "epsilon = 0"]
    step_sizes = [None, 0.1]
    names = ["non-stationary decreasing alpha", "non-stationary constant alpha = 0.1"]
    epsilon = 0.1
    for step_size in step_sizes:
        all_rewards = []
        all_actions = []
        all_max_rewards = []
        all_best_actions = []
        all_picked_correct_action = []
        while j < 2000:
            rewards = []
            actions = []
            max_reward = []
            best_action = []
            picked_correct_action = []
            while i < 10000:
                action, reward = bh.take_action(epsilon=epsilon, step_size=step_size, add_noise=True)
                rewards.append(reward)
                actions.append(action)
                max_reward.append(max(bh.q_star))
                best_action.append(np.array(bh.q_star).argmax())
                picked_correct_action.append(int(action == np.array(bh.q_star).argmax()))
                i += 1
            # if j % 20 == 0:
            #     print(".", end="")
            j += 1
            i = 0
            bh.reset()
            all_picked_correct_action.append(picked_correct_action)
            all_best_actions.append(best_action)
            all_max_rewards.append(max_reward)
            all_rewards.append(rewards)
            all_actions.append(actions)
        j = 0
        all_all_picked_correct_action.append(all_picked_correct_action)
        all_all_best_actions.append(all_best_actions)
        all_all_actions.append(all_actions)
        all_all_rewards.append(all_rewards)
        all_all_max_rewards.append(all_max_rewards)
    print("Plotting Rewards")
    plot_rewards(all_all_rewards, names, save_fig_dir, all_all_max_rewards)
    plot_actions(all_all_picked_correct_action, names, save_fig_dir)

    # a = 1
    # o = 0
    # for _ in range(100):
    #     o = some_function(o, a)
    #     print(a/o)

if __name__ == '__main__':
    main()