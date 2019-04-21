import numpy as np
from Environment import Environment
from DQNAgent import DQNAgent
from RandomAgent import  RandomAgent
import random

EPISODES = 1
if __name__ == "__main__":
    number = 11
    types = 4
    dimension = 20
    env = Environment(agents=number, dimension=dimension)
    state_size, action_size = env.get_state_action_size()
    print("state size", state_size, "action size", action_size)
    # agents with different types
    agents = []
    reward_history = []
    summary_history = []
    states = []
    actions = []
    cost_of_actions = []
    for i in range(types):
        agents.append([])
        reward_history.append([])
        states.append([])
        actions.append(np.zeros(number).astype(int))
        cost_of_actions.append(np.zeros(number).astype(int))
    for i in range(number):
        # agents apply DDQN-learning
        agents[0].append(DQNAgent(state_size, action_size, "DQN", 15))
        # agents apply DQN-learning
        agents[1].append(DQNAgent(state_size, action_size,"DDQN", 15))
        # agents randomly selection
        agents[2].append(RandomAgent(state_size, action_size))
        # agents apply LDDQN-learning
        agents[3].append(DQNAgent(state_size, action_size, "LDDQN", 15))

    env.reset_time()
    # env_state = [0 0 0 0 0 1 0 0 0 1 0]
    env_state = env.get_env_state()
    while env.time() < env.t_max:
        # at beginning
        if env.time() == 0:
            for agent_type in range(types):
                # take an action
                for agent_id in range(number):
                    actions[agent_type][agent_id] = (random.randrange(env.action_size))
                # observation an input
                utility_state = env.process_utility_state(actions[agent_type], env_state)
                for agent_id in range(number):
                    state_for_agent = env.process_state(actions[agent_type][agent_id], utility_state,
                                                        utility_state[actions[agent_type][agent_id]])
                    states[agent_type].append(state_for_agent)
        else:
            # each agent select an action, output = [2,4,6,7,...,8,9,3]
            for agent_type in range(types):
                # take an action
                for agent_id in range(number):
                    agents[agent_type][agent_id].switched = False
                    if agent_type == 2:
                        # handle random agents
                        if env_state[actions[agent_type][agent_id]] == 0:
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
                    else:
                        # handle DQN agents
                        action = agents[agent_type][agent_id].act(states[agent_type][agent_id])
                        if action != actions[agent_type][agent_id]:
                            cost_of_actions[agent_type][agent_id] += 1
                            agents[agent_type][agent_id].switched = True
                        actions[agent_type][agent_id] = action
        env.step()
        env_state = env.get_env_state()
        for agent_type in range(types):
            # take an action
            for agent_id in range(number):
                if agent_type == 2:
                    # handle agents randomly selection
                    reward = round(agents[agent_type][agent_id].reward(actions[agent_type], actions[agent_type][agent_id], env_state), 3)
                    #if agents[agent_type][agent_id].switched:
                        #reward = 1
                    agents[agent_type][agent_id].add_reward(reward)
                else:
                    # handle DQN agents
                    utility_state = env.process_utility_state(actions[agent_type], env_state)
                    reward = utility_state[actions[agent_type][agent_id]]
                    #if agents[agent_type][agent_id].switched:
                        #reward = 1
                    next_state_for_agent = env.process_state(actions[agent_type][agent_id], utility_state, reward)
                    agents[agent_type][agent_id].step(states[agent_type][agent_id], actions[agent_type][agent_id],
                                                       reward, next_state_for_agent, 0)
                    states[agent_type][agent_id] = next_state_for_agent
                    agents[agent_type][agent_id].add_reward(reward)
        total = []
        for agent_type in range(types):
            total.append(0)
            for agent_id in range(number):
                total[agent_type] += agents[agent_type][agent_id].reward_history[-1]
        if env.time() % 100 == 0:
            print(env.time(),
                  round(total[0] / (env.time() * number), 3),
                  round(total[1] / (env.time() * number), 3),
                  round(total[3] / (env.time() * number), 3),
                  round(total[2] / (env.time() * number), 3),
                  "Q=[", np.sum(cost_of_actions[0]), actions[0], "]",
                  "D=[", np.sum(cost_of_actions[1]), actions[1], "]",
                  "L=[", np.sum(cost_of_actions[3]), actions[3], "]",
                  "R=[", np.sum(cost_of_actions[2]), actions[2], "]",)
        # system total output
        if env.time() % 500 == 0:
            for agent_type in range(types):
                reward_history[agent_type].append(round(total[agent_type] / (env.time() * number), 3))
            summary_history.append(env.temp_summary(number))
            print("************************************")
            print(summary_history, reward_history)
            print("************************************")
    env.summary()
    print("************************************")
    print(reward_history)
    print("************************************")
