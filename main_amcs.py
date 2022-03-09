import gym
from agent import Sarsa_Agent
from gridworld import CliffWalkingWapper
import time



# Sarsa algo
def train_phase(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    state = env.reset()
    action = agent.sample(state)

    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = agent.sample(next_state)

        agent.learning(state, action, reward, next_state, next_action, done)  # Train the Sarsa brain

        action = next_action  # Update action from t to t+1
        state = next_state  # Update state from t to t+1

        total_reward += reward  # Accumulated reward
        total_steps += 1  # Record the step number
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_phase(env, agent):
    total_reward = 0
    state = env.reset()
    while True:
        action = agent.predict(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        time.sleep(.5)
        env.render()
        if done:
            print('Test reward = %.1f' % total_reward)
            break


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)

    agent = Sarsa_Agent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.1)

    is_render = False
    for episode_i in range(500):
        episo_reward, episo_steps = train_phase(env, agent, is_render)
        print('Episode %s: steps = %s, reward = %.1f'% (episode_i, episo_reward, episo_reward))

        # Render the env every 20 episodes
        if episode_i % 20 ==0:
            is_render = True
        else:
            is_render = False

    test_phase(env, agent)
