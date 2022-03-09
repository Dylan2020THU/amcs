import numpy as np

class Sarsa_Agent:
    def __init__(self, state_n, act_n, learning_rate=0.01, gamma=0.9, epsilon=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((state_n, act_n))  # Q table

    def sample(self, state):
        if np.random.uniform(0, 1) < (1 - self.epsilon):
            action = self.predict(state)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self, state):
        Q_list = self.Q[state, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action

    def learning(self, state, action, reward, next_state, next_action, done):
        predict_Q = self.Q[state, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] = self.Q[state, action] + self.lr * (target_Q - predict_Q)  # Update Q-value using Sarsa
