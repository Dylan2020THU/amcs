'''
Proj: Aircraft Maintenance Check Scheduling (AMCS) problem
Method: Reinforcement Learning
Latest update: 2022.3.9
'''

# import gym
import numpy as np
from agent import Sarsa_Agent
# from gridworld import CliffWalkingWapper
# import time
from env_AMCS import airport
import matplotlib.pyplot as plt



# Sarsa algo
def train_phase(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    # state = env.reset()
    state = env.airport_state()
    # print(state)
    action = agent.sample(state)

    while True:
        next_state, reward, done = env.step(action)
        next_action = agent.sample(next_state)

        agent.learning(state, action, reward, next_state, next_action, done)  # Train the Sarsa brain

        action = next_action  # Update action from t to t+1
        state = next_state  # Update state from t to t+1

        total_reward += reward  # Accumulated reward
        total_steps += 1  # Record the step number
        # if render:
        #     env.render()
        if done:
            break
    return total_reward, total_steps


def test_phase(env, agent):
    total_reward = 0
    state = env.airport_state()
    while True:
        action = agent.predict(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        # time.sleep(.5)
        # env.render()
        if done:
            print('Test reward = %.1f' % total_reward)
            break


if __name__ == '__main__':
    # env = gym.make("CliffWalking-v0")
    # env = CliffWalkingWapper(env)
    #
    # agent = Sarsa_Agent(
    #     state_n=env.observation_space.n,
    #     act_n=env.action_space.n,
    #     learning_rate=0.1,
    #     gamma=0.9,
    #     epsilon=0.1)
    #
    # is_render = False
    # for episode_i in range(500):
    #     episo_reward, episo_steps = train_phase(env, agent, is_render)
    #     print('Episode %s: steps = %s, reward = %.1f'% (episode_i, episo_reward, episo_reward))
    #
    #     # Render the env every 20 episodes
    #     if episode_i % 20 == 0:
    #         is_render = True
    #     else:
    #         is_render = False

    env = airport()
    print(env.airport_state)
    agent = Sarsa_Agent(
        state_n=4,
        act_n=4,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1)

    n_episode = 500
    accumulated_reward_list = np.zeros(n_episode)
    accumulated_reward = 0
    for episode_i in range(n_episode):
        episo_reward, episo_steps = train_phase(env, agent)
        accumulated_reward += episo_reward
        accumulated_reward_list[episode_i] = accumulated_reward
        print('Episode %s: steps = %s, reward = %.1f'% (episode_i, episo_reward, episo_reward))
    print('Accumulated reward list:',accumulated_reward_list)

    # Reward visualization
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(n_episode), accumulated_reward_list)
    ax.set(title='Accumulated Reward', ylabel='Reward', xlabel='Episode')
    plt.show()

    test_phase(env, agent)
