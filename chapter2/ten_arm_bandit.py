#!/usr/bin/env python
"""Run a 10 arm bandit problem using action-value methods"""
########################################################################
# File: ten_arm_bandit.py
#  executable: ten_arm_bandit.py
#
# Author: Andrew Bailey
# History: 02/05/18 Created
########################################################################

import abc
import os
import sys
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
from timeit import default_timer as timer


class Bandit(object):

    def __init__(self, n_bandits, q_star, stationary=False):
        self.n_bandits = n_bandits
        self.stationary = stationary
        if q_star is not None:
            self.q_star = q_star
        else:
            self.q_star = np.random.normal(loc=0, scale=1, size=self.n_bandits)
        self.original_q_star = self.q_star

    @staticmethod
    def calculate_running_average(original_average, step_size, new_value):
        """Calculate the new action value given step size, previous action and new reward"""
        return original_average + (step_size * (new_value-original_average))

    def reward_function(self, action_index):
        """Return the reward function for function where the distributions were selected via a random sampling of
        the standard normal and variance is 1"""
        return np.random.normal(loc=self.q_star[action_index], scale=1)

    def reset_q_star(self):
        self.q_star = self.original_q_star


class GradientBandit(Bandit):
    def __init__(self, n_bandits, q_star=None, stationary=False):
        super().__init__(n_bandits, q_star, stationary)
        self.action_preference = [0.0] * self.n_bandits
        self.average_reward = 0.0
        self.action_probs = [0.0] * self.n_bandits
        self.update_probabilities()

    def calculate_new_action_preferences(self, step_size, reward, action_index):
        """Calculate the new action preference given all necessary parameters
        pg 37 of RL sutton + barto
        """
        new_preferences = []
        for i, preference in enumerate(self.action_preference):
            if i == action_index:
                new_preferences.append(preference + (step_size * (reward - self.average_reward) * (1 - self.action_probs[i])))
            else:
                new_preferences.append(preference - (step_size * (reward - self.average_reward) * (self.action_probs[i])))
        self.action_preference = new_preferences

    def update_probabilities(self):
        self.action_probs = np.exp(self.action_preference)/sum(np.exp(self.action_preference))

    def select_action(self):
        return np.random.choice(range(self.n_bandits), p=self.action_probs)

    def take_action(self, step_size=None):
        action_index = self.select_action()
        reward = self.reward_function(action_index)

        self.calculate_new_action_preferences(step_size, reward, action_index)
        self.average_reward = self.calculate_running_average(self.average_reward, step_size, reward)
        self.update_probabilities()
        if not self.stationary:
            self.q_star = [self.q_star[i]+x for i, x in enumerate(np.random.normal(0, 0.01, self.n_bandits))]

        return action_index, reward

    def run_algorithm(self, iterations, step_size):
        rewards = []
        actions = []
        max_reward = []
        best_action = []
        picked_correct_action = []
        for i in range(iterations):

            action, reward = self.take_action(step_size=step_size)
            rewards.append(reward)
            actions.append(action)
            max_reward.append(max(self.q_star))
            best_action.append(np.array(self.q_star).argmax())
            picked_correct_action.append(int(action == np.array(self.q_star).argmax()))

        return rewards, actions, max_reward, best_action, picked_correct_action

    def reset(self):
        """Reset parameters for another iteration"""
        self.action_preference = [0] * self.n_bandits
        self.average_reward = 0.0
        self.action_probs = [0] * self.n_bandits
        self.update_probabilities()
        self.reset_q_star()


class UcbBandit(Bandit):
    """Upper confidence bound"""
    def __init__(self, n_bandits, q_star=None, stationary=False, q_t_initial_value=0):
        super().__init__(n_bandits, q_star, stationary)
        self.q_t_initial_value = q_t_initial_value
        self.q_t = [self.q_t_initial_value] * self.n_bandits
        self.n_steps = [0] * self.n_bandits
        self.total_steps = 0

    def select_action(self, c):
        nl_t = np.log(self.total_steps)
        calculation = np.array([self.q_t[x] + c * (np.sqrt(nl_t / self.n_steps[x])) for x in range(self.n_bandits)])
        return calculation.argmax()

    def take_action(self, c, step_size=None):
        """Take an action, get reward, update n_steps and update value function"""
        action_index = self.select_action(c)
        reward = self.reward_function(action_index)
        self.n_steps[action_index] += 1
        self.total_steps += 1
        if step_size is None:
            step_size = 1 / self.n_steps[action_index]

        self.q_t[action_index] = self.calculate_running_average(self.q_t[action_index], step_size, reward)
        if not self.stationary:
            self.q_star = [self.q_star[i]+x for i, x in enumerate(np.random.normal(0, 0.01, self.n_bandits))]

        return action_index, reward

    def reset(self):
        """Reset parameters for another iteration"""
        self.q_t = [self.q_t_initial_value] * self.n_bandits
        self.n_steps = [0] * self.n_bandits
        self.reset_q_star()

    def run_algorithm(self, iterations, c, step_size):
        rewards = []
        actions = []
        max_reward = []
        best_action = []
        picked_correct_action = []
        for i in range(iterations):

            action, reward = self.take_action(c=c, step_size=step_size)
            rewards.append(reward)
            actions.append(action)
            max_reward.append(max(self.q_star))
            best_action.append(np.array(self.q_star).argmax())
            picked_correct_action.append(int(action == np.array(self.q_star).argmax()))

        return rewards, actions, max_reward, best_action, picked_correct_action


class GreedyBandit(Bandit):

    def __init__(self, n_bandits, q_star=None, stationary=False, q_t_initial_value=0):
        super().__init__(n_bandits, q_star, stationary)

        self.q_t_initial_value = q_t_initial_value
        self.q_t = [self.q_t_initial_value] * self.n_bandits
        self.n_steps = [0] * self.n_bandits

    def select_action(self, epsilon=0):
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

    def take_action(self, epsilon, step_size=None):
        """Take an action, get reward, update n_steps and update value function"""
        action_index = self.select_action(epsilon)
        reward = self.reward_function(action_index)
        self.n_steps[action_index] += 1
        if step_size is None:
            step_size = 1 / self.n_steps[action_index]

        self.q_t[action_index] = self.calculate_running_average(self.q_t[action_index], step_size, reward)
        if not self.stationary:
            self.q_star = [self.q_star[i]+x for i, x in enumerate(np.random.normal(0, 0.01, self.n_bandits))]

        return action_index, reward

    def reset(self):
        """Reset parameters for another iteration"""
        self.q_t = [self.q_t_initial_value] * self.n_bandits
        self.n_steps = [0] * self.n_bandits
        self.reset_q_star()

    def run_algorithm(self, iterations, epsilon, step_size):
        rewards = []
        actions = []
        max_reward = []
        best_action = []
        picked_correct_action = []
        for i in range(iterations):

            action, reward = self.take_action(epsilon=epsilon, step_size=step_size)
            rewards.append(reward)
            actions.append(action)
            max_reward.append(max(self.q_star))
            best_action.append(np.array(self.q_star).argmax())
            picked_correct_action.append(int(action == np.array(self.q_star).argmax()))

        return rewards, actions, max_reward, best_action, picked_correct_action


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

    panel1.legend(loc='lower right', fancybox=True, shadow=True)
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
    panel1.legend(loc='lower right', fancybox=True, shadow=True)
    if save_fig_dir is not None:
        plt.savefig(os.path.join(save_fig_dir, "percent_correct_action.png"))
    else:
        plt.show()


def plot_bandit_algorithm_comparison(names, data, save_fig_dir=None):
    plt.figure(figsize=(7, 7))
    panel1 = plt.axes([0.1, 0.1, .8, .8], xscale="log")
    panel1.set_xlabel('Epsilon, Alpha, C, Q_0')
    panel1.set_ylabel('Average reward')
    panel1.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    # panel1.xaxis.set_major_locator(ticker.AutoLocator())
    # panel1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    panel1.xaxis.set_ticks([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3, 4])
    panel1.xaxis.set_ticklabels(["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "3", "4"])
    for name, reward in zip(names, data):
        panel1.plot(reward[0], reward[1], label=name)
    panel1.legend(loc='lower left', fancybox=True, shadow=True)
    if save_fig_dir is not None:
        plt.savefig(os.path.join(save_fig_dir, "bandit_algorithm_comparison.png"))
    else:
        plt.show()


def some_function(a=1, o=0):
    def some_function2(o, a):
        return o + a*(1 - o)

    for _ in range(100):
        o = some_function2(o, a)
        print(a/o)


def problem_2_5():
    n_bandits = 10
    save_fig_dir = '/'.join(os.path.abspath(__file__).split("/")[:-1])

    # save_fig_dir = os.path.join(HOME, "chapter2")
    # save_fig_dir = None
    np.random.normal(loc=0, scale=0.1)
    q_star = np.random.normal(loc=0, scale=1, size=n_bandits)
    e_g = GreedyBandit(n_bandits, q_star=q_star, stationary=False, q_t_initial_value=0)
    step_sizes = [None, 0.1]
    names = ["non-stationary decreasing alpha", "non-stationary constant alpha = 0.1"]
    ####
    epsilon = 0.1
    iterations = 1000
    n_repeats = 2000
    j = 0
    ####
    all_all_rewards = []
    all_all_actions = []
    all_all_best_actions = []
    all_all_max_rewards = []
    all_all_picked_correct_action = []
    ####
    for step_size in step_sizes:
        all_rewards = []
        all_actions = []
        all_max_rewards = []

        all_best_actions = []
        all_picked_correct_action = []
        while j < n_repeats:
            rewards, actions, max_reward, best_action, picked_correct_action = e_g.run_algorithm(iterations, epsilon,
                                                                                                 step_size)
            j += 1
            e_g.reset()
            ####
            all_picked_correct_action.append(picked_correct_action)
            all_best_actions.append(best_action)
            all_max_rewards.append(max_reward)
            all_rewards.append(rewards)
            all_actions.append(actions)
        j = 0
        e_g.reset()
        ####
        all_all_picked_correct_action.append(all_picked_correct_action)
        all_all_best_actions.append(all_best_actions)
        all_all_actions.append(all_actions)
        all_all_rewards.append(all_rewards)
        all_all_max_rewards.append(all_max_rewards)
    print("Plotting Rewards")
    plot_rewards(all_all_rewards, names, save_fig_dir, all_all_max_rewards)
    plot_actions(all_all_picked_correct_action, names, save_fig_dir)


def problem_2_11():
    n_bandits = 10
    iterations = 1000
    repeats = 200
    save_fig_dir = "/Users/andrewbailey/CLionProjects/ReinforcementLearningHomework/chapter2"
    # save_fig_dir = None
    stationary = False
    q_star = np.random.normal(loc=0, scale=1, size=n_bandits)
    all_all_max_rewards = []
    all_all_average_rewards = []
    names = ["epsilon greedy", "greedy with optimistic initialization\nalpha = 0.1", "gradient learning", "UCB"]
    # epsilon Greedy
    all_steps = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3]

    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    step_size = 0.1

    print("greedy")
    all_max_rewards = []
    all_average_rewards = []
    for epsilon in epsilons:
        average_rewards = []
        max_rewards = []
        picked_correct_actions = []
        for i in range(repeats):
            e_g = GreedyBandit(n_bandits, q_star=q_star, stationary=stationary, q_t_initial_value=0)

            rewards, actions, max_reward, best_action, picked_correct_action = e_g.run_algorithm(iterations, epsilon,
                                                                                                 step_size)
            average_rewards.append(np.mean(rewards))
            max_rewards.append(np.max(max_reward))
            picked_correct_actions.append(np.mean(picked_correct_action))
        print(np.mean(picked_correct_actions))
        all_max_rewards.append(np.mean(max_rewards))
        all_average_rewards.append(np.mean(average_rewards))
    all_all_average_rewards.append([epsilons, all_average_rewards])
    all_all_max_rewards.append(all_max_rewards)

    print("optimistic")
    # greedy with optimistic initialization
    initalization_values = [1/4, 1/2, 1, 2, 4]
    step_size = 0.1
    all_max_rewards = []
    all_average_rewards = []
    for q_0 in all_steps:
        picked_correct_actions = []
        average_rewards = []
        max_rewards = []
        for i in range(repeats):
            e_g = GreedyBandit(n_bandits, q_star=q_star, stationary=stationary, q_t_initial_value=q_0)
            rewards, actions, max_reward, best_action, picked_correct_action = e_g.run_algorithm(iterations, 0,
                                                                                                 step_size)
            average_rewards.append(np.mean(rewards))
            max_rewards.append(np.max(max_reward))
            picked_correct_actions.append(np.mean(picked_correct_action))
        print(np.mean(picked_correct_actions))
        all_max_rewards.append(np.mean(max_rewards))
        all_average_rewards.append(np.mean(average_rewards))
    all_all_average_rewards.append([all_steps, all_average_rewards])
    all_all_max_rewards.append(all_max_rewards)

    print("gradient")
    # gradient learning
    step_sizes = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3]
    all_max_rewards = []
    all_average_rewards = []
    for step_size in all_steps:
        picked_correct_actions = []

        average_rewards = []
        max_rewards = []
        for i in range(repeats):
            e_g = GradientBandit(n_bandits, q_star=q_star, stationary=stationary)
            rewards, actions, max_reward, best_action, picked_correct_action = e_g.run_algorithm(iterations, step_size)
            average_rewards.append(np.mean(rewards))
            max_rewards.append(np.max(max_reward))
            picked_correct_actions.append(np.mean(picked_correct_action))
        print(np.mean(picked_correct_actions))
        all_max_rewards.append(np.mean(max_rewards))
        all_average_rewards.append(np.mean(average_rewards))
    all_all_average_rewards.append([all_steps, all_average_rewards])
    all_all_max_rewards.append(all_max_rewards)

    # UCB learning
    print("UCB")
    cs = [1/16, 1/8, 1/4, 1/2, 1, 2, 3, 4]
    step_size = 0.1
    all_max_rewards = []
    all_average_rewards = []
    for c in all_steps:
        picked_correct_actions = []

        average_rewards = []
        max_rewards = []
        for i in range(repeats):
            e_g = UcbBandit(n_bandits, q_star=q_star, stationary=stationary)
            rewards, actions, max_reward, best_action, picked_correct_action = e_g.run_algorithm(iterations, c,
                                                                                                 step_size)
            average_rewards.append(np.mean(rewards))
            max_rewards.append(np.max(max_reward))
            picked_correct_actions.append(np.mean(picked_correct_action))
        print(np.mean(picked_correct_actions))
        all_max_rewards.append(np.mean(max_rewards))
        all_average_rewards.append(np.mean(average_rewards))
    all_all_average_rewards.append([all_steps, all_average_rewards])
    all_all_max_rewards.append(all_max_rewards)

    print("Plotting Rewards")
    plot_bandit_algorithm_comparison(names, all_all_average_rewards, save_fig_dir=save_fig_dir)



if __name__ == '__main__':
    start = timer()
    # mutltiprocess_problem_2_11()
    problem_2_11()
    # problem_2_5()
    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)

