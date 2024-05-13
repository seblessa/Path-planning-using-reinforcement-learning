from .metrics import print_metrics
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN
from .environment import Environment
from .utils import latest_model


def test_model1(algorithm, algo_name):
    env = Environment()
    model_path = latest_model(algo_name)
    print(f"Testing: {model_path}")
    model = algorithm.load(model_path, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, warn=False)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test_model(algorithm, algo_name):
    env = Environment()
    model_path = latest_model(algo_name)
    model = algorithm.load(model_path, env=env)

    num_wins = 0
    episodes = 10
    goal_position = env.goal_position
    width_object = env.goal_distance * 2
    metrics_info = {"num_episodes": episodes, "num_successful_trials": num_wins, "time_to_reach_goal": 0,
                    "energy_consumption": 0, "distance_to_goal": 0}

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, win, info = env.step(action.item())
            if win:
                num_wins += 1
                print("Goal Reached")


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
