import gymnasium as gym
import numpy as np



## Define class to wrap PointMaze environment with a custom reward function
class PointMazeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.total_obs = self.reset()
        
    def step(self, action):
        self.total_obs = self.env.step(action)
        self.goal = self.total_obs[0]["desired_goal"]
        self.obs = np.append(self.total_obs[0]["observation"][:2], self.total_obs[0]["desired_goal"])
        # print(obs)
        done = self.total_obs[4]["success"]
        reward = self._reward(self.obs, action)
        
        return self.obs, reward, done, {}
    def reset(self):
        self.total_obs = self.env.reset()
        self.goal = self.total_obs[0]["desired_goal"]
        self.obs = np.append(self.total_obs[0]["observation"][:2], self.total_obs[0]["desired_goal"])
        return self.obs

    def _reward(self, obs, action):
        reward = -np.linalg.norm(obs[:2] - self.goal) - 0.1 * np.linalg.norm(action)
        return reward