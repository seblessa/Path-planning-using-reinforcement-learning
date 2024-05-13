from .environment import Environment
from .metrics import success_rate, time_to_reach_goal, distance_to_goal, energy_consumption
from stable_baselines3 import PPO
import os


# TODO: Implement metrics

def run(model_path: str = None, goal_position: (float, float) = (0,0)):
    env = Environment()
    env.reset()

    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'models/PPO', '40000.zip')

    model = PPO.load(model_path, env=env)

    num_wins = 0
    episodes = 10
    goal_position = env.goal_position
    width_object = env.goal_distance * 2

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, win, info = env.step(action.item())
