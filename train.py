import gymnasium as gym
import wandb
import random
from agent import RandomAgent, SARSAAgent

# enable/disable wandb with the environment variable WANDB_DISABLED
import os
os.environ["WANDB_DISABLED"] = "true"

from env import PointMazeWrapper




episodes = 1000
steps = 1000
learning_rate = 0.02
alpha = 0.05
gamma = 0.99

wandb.init(
    project="maze_rl",
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "SARSA",
    "episodes": episodes,
    "steps": steps,
    "env": "PointMaze_UMazeDense-v3",
    "alpha": alpha,
    "gamma": gamma,
    }
)

env = PointMazeWrapper(gym.make('PointMaze_UMazeDense-v3', render_mode='human'))
env.reset()

agent = SARSAAgent(env)
try:
    agent.load("./q_table.npy")
    print(f"Loaded q_table: {agent.q}")
    print(f"{agent.q.sum()}")
except:
    print("No q_table found, starting from scratch")
    pass

for episode in range(episodes):
    for step in range(steps):
        done = agent.step()
        if done or step == steps - 1:
            agent.reset()
            continue
    wandb.log({"total_reward": agent.total_reward, "episode": episode})
    agent.save("q_table.npy")
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()