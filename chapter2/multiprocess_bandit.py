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

from ten_arm_bandit import *
from signalalign.utils import multithread


def multiprocess_ucb(iterations, repeats, cs, q_star, worker_count=1, n_bandits=10, stationary=True, step_size=0.1):
    """Multiprocess for filtering reads but dont move the files
    :param worker_count: number of workers to use
    :param debug: boolean option which will only use one process in order to fail if an error arises
    :return: True
    """
    filter_reads_args = {"n_bandits": n_bandits, "stationary": stationary, "iterations": iterations, "q_star": q_star,
                         "repeats": repeats, "step_size": step_size}
    total, failure, messages, output = multithread.run_service2(
        ucb_bandit_service, [c for c in cs],
        filter_reads_args, ["c"], worker_count)
    output.sort(key=lambda x: x[3])
    return output


def ucb_bandit_service(work_queue, done_queue, service_name="ucb_bandit_service"):
    """
    :param work_queue: arguments to be done
    :param done_queue: errors and returns to be put
    :param service_name: name of the service
    """
    # prep
    total_handled = 0
    failure_count = 0
    mem_usages = list()

    # catch overall exceptions
    try:
        for f in iter(work_queue.get, 'STOP'):
            # catch exceptions on each element
            try:
                reads = ucb_bandit_wrapper(**f)
                done_queue.put(reads)
            except Exception as e:
                # get error and log it
                message = "{}:{}".format(type(e), str(e))
                error = "{} '{}' failed with: {}".format(service_name, multithread.current_process().name, message)
                print("[{}] ".format(service_name) + error)
                done_queue.put(error)
                failure_count += 1

            # increment total handling
            total_handled += 1

    except Exception as e:
        # get error and log it
        message = "{}:{}".format(type(e), str(e))
        error = "{} '{}' critically failed with: {}".format(service_name, multithread.current_process().name, message)
        print("[{}] ".format(service_name) + error)
        done_queue.put(error)

    finally:
        # logging and final reporting
        print("[%s] '%s' completed %d calls with %d failures"
              % (service_name, multithread.current_process().name, total_handled, failure_count))
        done_queue.put("{}:{}".format(multithread.TOTAL_KEY, total_handled))
        done_queue.put("{}:{}".format(multithread.FAILURE_KEY, failure_count))
        if len(mem_usages) > 0:
            done_queue.put("{}:{}".format(multithread.MEM_USAGE_KEY, ",".join(map(str, mem_usages))))


def ucb_bandit_wrapper(n_bandits, c, q_star, stationary, iterations, repeats, step_size=0.1):
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

    return np.mean(average_rewards), np.mean(max_rewards), np.mean(picked_correct_actions), c


def gradient_bandit_wrapper(n_bandits, step_size, q_star, stationary, iterations, repeats):
    picked_correct_actions = []

    average_rewards = []
    max_rewards = []
    for i in range(repeats):
        e_g = GradientBandit(n_bandits, q_star=q_star, stationary=stationary)
        rewards, actions, max_reward, best_action, picked_correct_action = e_g.run_algorithm(iterations, step_size)
        average_rewards.append(np.mean(rewards))
        max_rewards.append(np.max(max_reward))
        picked_correct_actions.append(np.mean(picked_correct_action))

    return np.mean(average_rewards), np.mean(max_rewards), np.mean(picked_correct_actions), step_size


def multiprocess_gradient_bandit(iterations, repeats, step_sizes, q_star, worker_count=1, n_bandits=10, stationary=True):
    """Multiprocess for filtering reads but dont move the files
    :param worker_count: number of workers to use
    :param debug: boolean option which will only use one process in order to fail if an error arises
    :return: True
    """
    filter_reads_args = {"n_bandits": n_bandits, "stationary": stationary, "iterations": iterations, "q_star": q_star,
                         "repeats": repeats}
    total, failure, messages, output = multithread.run_service2(
        gradient_bandit_service, [c for c in step_sizes],
        filter_reads_args, ["step_size"], worker_count)
    output.sort(key=lambda x: x[3])
    return output


def gradient_bandit_service(work_queue, done_queue, service_name="ucb_bandit_service"):
    """
    :param work_queue: arguments to be done
    :param done_queue: errors and returns to be put
    :param service_name: name of the service
    """
    # prep
    total_handled = 0
    failure_count = 0
    mem_usages = list()

    # catch overall exceptions
    try:
        for f in iter(work_queue.get, 'STOP'):
            # catch exceptions on each element
            try:
                reads = gradient_bandit_wrapper(**f)
                done_queue.put(reads)
            except Exception as e:
                # get error and log it
                message = "{}:{}".format(type(e), str(e))
                error = "{} '{}' failed with: {}".format(service_name, multithread.current_process().name, message)
                print("[{}] ".format(service_name) + error)
                done_queue.put(error)
                failure_count += 1

            # increment total handling
            total_handled += 1

    except Exception as e:
        # get error and log it
        message = "{}:{}".format(type(e), str(e))
        error = "{} '{}' critically failed with: {}".format(service_name, multithread.current_process().name, message)
        print("[{}] ".format(service_name) + error)
        done_queue.put(error)

    finally:
        # logging and final reporting
        print("[%s] '%s' completed %d calls with %d failures"
              % (service_name, multithread.current_process().name, total_handled, failure_count))
        done_queue.put("{}:{}".format(multithread.TOTAL_KEY, total_handled))
        done_queue.put("{}:{}".format(multithread.FAILURE_KEY, failure_count))
        if len(mem_usages) > 0:
            done_queue.put("{}:{}".format(multithread.MEM_USAGE_KEY, ",".join(map(str, mem_usages))))


def epsilon_greedy_bandit_wrapper(n_bandits, epsilon, q_star, stationary, iterations, repeats, step_size=0.1):
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
    return np.mean(average_rewards), np.mean(max_rewards), np.mean(picked_correct_actions), epsilon


def multiprocess_epsilon_greedy_bandit(iterations, repeats, epsilons, q_star, worker_count=1, n_bandits=10, stationary=True, step_size=0.1):
    """Multiprocess for filtering reads but dont move the files
    :param worker_count: number of workers to use
    :param debug: boolean option which will only use one process in order to fail if an error arises
    :return: True
    """
    filter_reads_args = {"n_bandits": n_bandits, "stationary": stationary, "iterations": iterations, "q_star": q_star,
                         "repeats": repeats, 'step_size': step_size}
    total, failure, messages, output = multithread.run_service2(
        epsilon_greedy_bandit_service, epsilons,
        filter_reads_args, ["epsilon"], worker_count)
    output.sort(key=lambda x: x[3])
    return output


def epsilon_greedy_bandit_service(work_queue, done_queue, service_name="epsilon_greedy_bandit_service"):
    """
    :param work_queue: arguments to be done
    :param done_queue: errors and returns to be put
    :param service_name: name of the service
    """
    # prep
    total_handled = 0
    failure_count = 0
    mem_usages = list()

    # catch overall exceptions
    try:
        for f in iter(work_queue.get, 'STOP'):
            # catch exceptions on each element
            try:
                reads = epsilon_greedy_bandit_wrapper(**f)
                done_queue.put(reads)
            except Exception as e:
                # get error and log it
                message = "{}:{}".format(type(e), str(e))
                error = "{} '{}' failed with: {}".format(service_name, multithread.current_process().name, message)
                print("[{}] ".format(service_name) + error)
                done_queue.put(error)
                failure_count += 1

            # increment total handling
            total_handled += 1

    except Exception as e:
        # get error and log it
        message = "{}:{}".format(type(e), str(e))
        error = "{} '{}' critically failed with: {}".format(service_name, multithread.current_process().name, message)
        print("[{}] ".format(service_name) + error)
        done_queue.put(error)

    finally:
        # logging and final reporting
        print("[%s] '%s' completed %d calls with %d failures"
              % (service_name, multithread.current_process().name, total_handled, failure_count))
        done_queue.put("{}:{}".format(multithread.TOTAL_KEY, total_handled))
        done_queue.put("{}:{}".format(multithread.FAILURE_KEY, failure_count))
        if len(mem_usages) > 0:
            done_queue.put("{}:{}".format(multithread.MEM_USAGE_KEY, ",".join(map(str, mem_usages))))

def optimistic_bandit_wrapper(n_bandits, q_0, q_star, stationary, iterations, repeats, step_size=0.1):
    average_rewards = []
    max_rewards = []
    picked_correct_actions = []
    for i in range(repeats):
        e_g = GreedyBandit(n_bandits, q_star=q_star, stationary=stationary, q_t_initial_value=q_0)

        rewards, actions, max_reward, best_action, picked_correct_action = e_g.run_algorithm(iterations, 0,
                                                                                             step_size)
        average_rewards.append(np.mean(rewards))
        max_rewards.append(np.max(max_reward))
        picked_correct_actions.append(np.mean(picked_correct_action))
    return np.mean(average_rewards), np.mean(max_rewards), np.mean(picked_correct_actions), q_0


def multiprocess_optimistic_bandit(iterations, repeats, init_values, q_star, worker_count=1, n_bandits=10, stationary=True, step_size=0.1):
    """Multiprocess for filtering reads but dont move the files
    :param worker_count: number of workers to use
    :param debug: boolean option which will only use one process in order to fail if an error arises
    :return: True
    """
    filter_reads_args = {"n_bandits": n_bandits, "stationary": stationary, "iterations": iterations, "q_star": q_star,
                         "repeats": repeats, 'step_size': step_size}
    total, failure, messages, output = multithread.run_service2(
        optimistic_bandit_service, init_values,
        filter_reads_args, ["q_0"], worker_count)
    output.sort(key=lambda x: x[3])
    return output


def optimistic_bandit_service(work_queue, done_queue, service_name="optimistic_bandit_service"):
    """
    :param work_queue: arguments to be done
    :param done_queue: errors and returns to be put
    :param service_name: name of the service
    """
    # prep
    total_handled = 0
    failure_count = 0
    mem_usages = list()

    # catch overall exceptions
    try:
        for f in iter(work_queue.get, 'STOP'):
            # catch exceptions on each element
            try:
                reads = optimistic_bandit_wrapper(**f)
                done_queue.put(reads)
            except Exception as e:
                # get error and log it
                message = "{}:{}".format(type(e), str(e))
                error = "{} '{}' failed with: {}".format(service_name, multithread.current_process().name, message)
                print("[{}] ".format(service_name) + error)
                done_queue.put(error)
                failure_count += 1

            # increment total handling
            total_handled += 1

    except Exception as e:
        # get error and log it
        message = "{}:{}".format(type(e), str(e))
        error = "{} '{}' critically failed with: {}".format(service_name, multithread.current_process().name, message)
        print("[{}] ".format(service_name) + error)
        done_queue.put(error)

    finally:
        # logging and final reporting
        print("[%s] '%s' completed %d calls with %d failures"
              % (service_name, multithread.current_process().name, total_handled, failure_count))
        done_queue.put("{}:{}".format(multithread.TOTAL_KEY, total_handled))
        done_queue.put("{}:{}".format(multithread.FAILURE_KEY, failure_count))
        if len(mem_usages) > 0:
            done_queue.put("{}:{}".format(multithread.MEM_USAGE_KEY, ",".join(map(str, mem_usages))))


def mutltiprocess_problem_2_11():
    n_bandits = 10
    iterations = 200000
    repeats = 200
    workers = 8
    save_fig_dir = "/Users/andrewbailey/CLionProjects/ReinforcementLearningHomework/chapter2"
    # save_fig_dir = None
    stationary = False
    q_star = np.random.normal(loc=0, scale=1, size=n_bandits)
    all_all_max_rewards = []
    all_all_average_rewards = []
    names = ["epsilon greedy", "greedy with optimistic initialization\nalpha = 0.1", "gradient learning", "UCB"]
    all_steps = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3]

    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    step_size = 0.1

    # epsilon Greedy
    print("greedy")
    data = multiprocess_epsilon_greedy_bandit(iterations, repeats, epsilons, q_star, stationary=stationary,
                                              worker_count=workers)

    all_all_average_rewards.append([[x[3] for x in data],  [x[0] for x in data]])
    all_all_max_rewards.append([x[2] for x in data])
    print([x[2] for x in data])
    print("optimistic")

    data = multiprocess_optimistic_bandit(iterations, repeats, all_steps, q_star, stationary=stationary,
                                          worker_count=workers)

    all_all_average_rewards.append([[x[3] for x in data],  [x[0] for x in data]])
    all_all_max_rewards.append([x[2] for x in data])
    print([x[2] for x in data])
    print("gradient")

    data = multiprocess_gradient_bandit(iterations, repeats, all_steps, q_star, stationary=stationary,
                                        worker_count=workers)
    all_all_average_rewards.append([[x[3] for x in data],  [x[0] for x in data]])
    all_all_max_rewards.append([x[2] for x in data])
    print([x[2] for x in data])
    print("UCB")

    data = multiprocess_ucb(iterations, repeats, all_steps, q_star, stationary=stationary, step_size=step_size,
                            worker_count=workers)
    all_all_average_rewards.append([[x[3] for x in data],  [x[0] for x in data]])
    all_all_max_rewards.append([x[2] for x in data])
    print([x[2] for x in data])

    print("Plotting Rewards")
    plot_bandit_algorithm_comparison(names, all_all_average_rewards, save_fig_dir=save_fig_dir)


def main():
    workers = 2
    n_bandits = 10
    iterations = 2000
    repeats = 10
    stationary = True
    step_size = 0.1
    all_steps = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3]

    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    q_star = np.random.normal(loc=0, scale=1, size=n_bandits)

    # data = multiprocess_ucb(iterations, repeats, all_steps, q_star, stationary=stationary, step_size=step_size,
    #                                              worker_count=workers)
    # data = multiprocess_gradient_bandit(iterations, repeats, all_steps, q_star, stationary=stationary,
    #                                     worker_count=workers)
    # data = multiprocess_epsilon_greedy_bandit(iterations, repeats, epsilons, q_star, stationary=stationary,
    #                                             worker_count=workers)
    # data = multiprocess_optimistic_bandit(iterations, repeats, epsilons, q_star, stationary=stationary,
    #                                           worker_count=workers)

    # print(data)


if __name__ == '__main__':
    start = timer()
    mutltiprocess_problem_2_11()
    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)
