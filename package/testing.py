from .metrics import print_metrics
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN
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
    total_time = 0
    energy_consumption = 0
    position = (0, 0)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, win, info = env.step(action.item())
            if done:
                print("Episode finished")
            if win:
                num_wins += 1
                total_time += info["time"]
                position += info["gps_readings"]
                energy_consumption += info["battery"]
                print("Goal Reached")

    metrics_info = {"num_episodes": episodes, "num_successful_trials": num_wins, "time_to_reach_goal": total_time,
                    "energy_consumption": energy_consumption, "distance_to_goal": position,
                    "goal_position": goal_position,
                    "width_object": width_object}

    print_metrics(metrics_info)

def main(algo_name=None):
    if algo_name == "PPO":
        print("Testing PPO")
        test_model(PPO, algo_name)
    elif algo_name == "A2C":
        print("Testing A2C")
        test_model(A2C, algo_name)
    elif algo_name == "DQN":
        print("Testing DQN")
        test_model(DQN, algo_name)
