from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    def __init__(self, env):
        self.env = env
        self.state = env.reset()
        self.total_reward = 0.0
    
    @abstractmethod
    def step(self):
        raise NotImplementedError

    def reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0


class RandomAgent(Agent):
    def step(self):
        self.state, reward, done, _ = self.env.step(self.env.action_space.sample())
        self.total_reward += reward
        if done:
            self.reset()
        return done
    
class SARSAAgent(Agent):
    def __init__(self, env, gamma=0.99, alpha=0.01):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 0.8
        # discretize observation and action space
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        self.discrete_obs_space = np.linspace(-5, 5, 10)
        self.discrete_act_space = np.linspace(env.action_space.low, env.action_space.high, 10)
        self.q = np.zeros((10, 10, 10, 10))
        print("Q shape:", self.q.shape)
        self.action = self.pick_action(self.state)

    def update_q(self, state, action, reward, next_state, next_action):
        state = state[0]["observation"][:2]
        state = np.digitize(state, self.discrete_obs_space)
        next_state = next_state[0]["observation"][:2]
        
        next_state = np.digitize(next_state, self.discrete_obs_space)
        print(f"state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, next_action: {next_action}")
        print(self.q[state[0]][state[1]][action[0]][action[1]])
        self.q[state[0]][state[1]][action[0]][action[1]] = self.q[state[0]][state[1]][action[0]][action[1]] + self.alpha * (reward + self.gamma * self.q[next_state[0]][next_state[1]][next_action[0]][next_action[1]] - self.q[state[0]][state[1]][action[0]][action[1]])

    def pick_action(self, state, update_q=False):
        if np.random.random() < self.epsilon and not update_q:
            self.epsilon *= 0.999
            # return random tuple of actions
            return np.random.randint(0, 10, size=2)
        state = state[0]["observation"][:2]
        state = np.digitize(state, self.discrete_obs_space)
        # get coordinates of index of maximum value in the q tensor
        max_action = np.unravel_index(np.argmax(self.q[state[0]][state[1]]), self.q[state[0]][state[1]].shape)
        # get row and column of maximum value in the q tensor
        # max_action = np.unravel_index(np.argmax(self.q[state]), self.q[state].shape)
        print("max_action:", max_action)

        return np.array(max_action)

    def step(self):
        prev_action = self.action
        # undigitize action
        self.action = [self.discrete_act_space[self.action[0]][0], self.discrete_act_space[self.action[1]][0]]
        print("action:", self.action)
        next_state, reward, done, _ = self.env.step(self.action)
        self.action = self.pick_action(next_state)
        self.update_q(self.state, prev_action, reward, next_state, self.action)
        self.total_reward += reward
        if done:
            self.reset()
        return done
    
    def save(self, filename):
        np.save(filename, self.q)

    def load(self, filename):
        self.q = np.load(filename)