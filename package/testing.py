from .metrics import success_rate, time_to_reach_goal, distance_to_goal, energy_consumption
from stable_baselines3 import PPO, SAC, DDPG, DQN
from .environment import Environment
from .utils import latest_model


def test_model(algorithm, algo_name):
    env = Environment()
    model_path = latest_model(algo_name)
    model = algorithm.load(model_path, env=env)

    num_wins = 0
    episodes = 1
    goal_position = env.goal_position
    width_object = env.goal_distance * 2

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, win, info = env.step(action.item())
        if win:
            print("Goal Reached")


def main(algo_name=None):
    if algo_name == "PPO":
        print("Testing PPO")
        test_model(PPO, algo_name)
    elif algo_name == "SAC":
        print("Testing SAC")
        test_model(SAC, algo_name)
    elif algo_name == "DDPG":
        print("Testing DDPG")
        test_model(DDPG, algo_name)
    elif algo_name == "DQN":
        print("Testing DQN")
        test_model(DQN, algo_name)


if __name__ == "__main__":
    main("PPO")
