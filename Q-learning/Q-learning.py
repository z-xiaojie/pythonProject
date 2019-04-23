import numpy as np
from Environment import Environment
from Agent import agent_create
import random
import time
import torch
import matplotlib.pyplot as plt


EPISODES = 1
if __name__ == "__main__":
    number = 10
    labels = ["DQN", "DDQN", "RANDOM"]
    types = 3
    dimension = 8
    env = Environment(agents=number, dimension=dimension, migration_overhead=4, data="new", service="diff")
    env.summary()
    state_size, action_size = env.get_state_action_size()
    print("state size", state_size, "action size", action_size)
    # agents with different types
    usage = []
    agents = []
    reward_history = []
    summary_history = []
    states = []
    actions = []
    cost_of_actions = []
    for agent_type in range(types):
        usage.append(np.zeros(dimension))
        agents.append([])
        reward_history.append([])
        states.append([])
        actions.append(np.zeros(number).astype(int))
        cost_of_actions.append(np.zeros(number).astype(int))
        for agent_id in range(number):
            agents[agent_type].append(agent_create(state_size, action_size, labels[agent_type]))
    env.reset_time()
    # env_state = [0 0 0 0 0 1 0 0 0 1 0]
    time_action = 0
    time_reward = 0
    env_state = env.get_env_state()
    while env.time() < env.t_max:
        """
             Take action
        """
        start = time.time()
        # at beginning
        if env.time() == 0:
            for agent_type in range(types):
                # take an action
                for agent_id in range(number):
                    actions[agent_type][agent_id] = (random.randrange(env.action_size))
                # observation an input
                utility_state = env.process_utility_state(agents[agent_type], actions[agent_type], env_state)
                for agent_id in range(number):
                    state_for_agent = env.process_state(agents[agent_type][agent_id].switched,
                                                        agents[agent_type][agent_id].migration_overhead,
                                                        actions[agent_type][agent_id], utility_state,
                                                        utility_state[actions[agent_type][agent_id]])
                    states[agent_type].append(state_for_agent)
        else:
            # each agent select an action, output = [2,4,6,7,...,8,9,3]
            for agent_type in range(types):
                # take an action
                for agent_id in range(number):
                    if agents[agent_type][agent_id].migration_overhead == 0:
                        agents[agent_type][agent_id].switched = False
                    if labels[agent_type] == "RANDOM":
                        # handle random agents
                        if env_state[actions[agent_type][agent_id]] == 0 or agents[agent_type][agent_id].migration_overhead > 0:
                            continue
                        else:
                            valid = []
                            for r in range(env.action_size):
                                if env_state[r] == 0:
                                    valid.append(r)
                            if len(valid) != 0:
                                actions[agent_type][agent_id] = valid[np.random.randint(len(valid), size=1)[0]]
                                cost_of_actions[agent_type][agent_id] += 1
                                agents[agent_type][agent_id].switched = True
                                agents[agent_type][agent_id].migration_overhead = env.migration_overhead
                    else:
                        if agents[agent_type][agent_id].migration_overhead == 0:
                            # handle DQN agents
                            action = agents[agent_type][agent_id].act(states[agent_type][agent_id])
                            if action != actions[agent_type][agent_id]:
                                cost_of_actions[agent_type][agent_id] += 1
                                agents[agent_type][agent_id].switched = True
                                agents[agent_type][agent_id].migration_overhead = env.migration_overhead
                            actions[agent_type][agent_id] = action
        time_action += time.time() - start
        """
            Update usage
        """
        for agent_type in range(types):
            for agent_id in range(number):
                usage[agent_type][actions[agent_type][agent_id]] += 1
        """
            Get reward
        """
        #c_2_reward = []
        start = time.time()
        env.step()
        env_state = env.get_env_state()
        for agent_type in range(types):
            # take an action
            utility_state = env.process_utility_state(agents[agent_type], actions[agent_type], env_state)
            for agent_id in range(number):
                if labels[agent_type] == "RANDOM":
                    # handle agents randomly selection
                    reward = utility_state[actions[agent_type][agent_id]]#round(agents[agent_type][agent_id].reward(actions[agent_type], actions[agent_type][agent_id], env_state), 3)
                    if agents[agent_type][agent_id].switched:
                        reward = 0
                        agents[agent_type][agent_id].migration_overhead -= 1
                    agents[agent_type][agent_id].add_reward(reward)
                    #c_2_reward.append(reward)
                else:
                    # handle DQN agents
                    reward = utility_state[actions[agent_type][agent_id]]
                    if agents[agent_type][agent_id].switched:
                        reward = 0
                        agents[agent_type][agent_id].migration_overhead -= 1
                    next_state_for_agent = env.process_state(agents[agent_type][agent_id].switched,
                                                             agents[agent_type][agent_id].migration_overhead,
                                                             actions[agent_type][agent_id], utility_state, reward)
                    agents[agent_type][agent_id].step(states[agent_type][agent_id], actions[agent_type][agent_id],
                                                      reward, next_state_for_agent, 0)
                    states[agent_type][agent_id] = next_state_for_agent
                    agents[agent_type][agent_id].add_reward(reward)
        time_reward += time.time() - start
        total = []
        for agent_type in range(types):
            total.append(0)
            for agent_id in range(number):
                total[agent_type] += agents[agent_type][agent_id].reward_history[-1]
        #print(env.time(),  "R=[", actions[2], "]", env_state)
        if env.time() % 500 == 0:
            current_optimal = env.temp_summary(number)
            summary = []
            for agent_type in range(types):
                ratio = total[agent_type] / current_optimal
                summary.append(round(ratio, 3))
            print(env.time(), "MAX(", current_optimal, ")[", summary[0], summary[1], summary[2], "]",
                  "Q=[", int(total[0]), actions[0], np.sum(cost_of_actions[0]), "]",
                  "D=[", int(total[1]), actions[1], np.sum(cost_of_actions[1]), "]",
                  "R=[", int(total[2]), actions[2], np.sum(cost_of_actions[2]), "]", env_state)
            # system total output
            if env.time() % 2000 == 0:
                for agent_type in range(types):
                    reward_history[agent_type].append(summary[agent_type])
                summary_history.append( round(current_optimal/ ( env.time() * number), 3))
                print("************************************")
                print(time_action, time_reward)
                print(summary_history)
                print(reward_history)
                print(usage)
                print("************************************")
                time_action = 0
                time_reward = 0

#   0   1   2   3   4   5   6   7
# [0.8 0.2 0.4 0.6 0.1 0.8 0.8 0.6]

