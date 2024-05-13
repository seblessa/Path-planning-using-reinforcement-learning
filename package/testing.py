from .metrics import success_rate, time_to_reach_goal, distance_to_goal, energy_consumption
from .environment import Environment
from .utils import latest_model
import sys


def test_model(algorithm):
    env = Environment()
    model = latest_model(algorithm)

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


def main(algo_name=None):
    if algo_name == "PPO":
        test_model(algo_name)
    if algo_name == "SAC":
        test_model(algo_name)
    elif algo_name == "DDPG":
        test_model(algo_name)
    elif algo_name == "DQN":
        test_model(algo_name)

    else:
        print("Invalid algorithm.")
        sys.exit(1)


if __name__ == "__main__":
    main("PPO")
