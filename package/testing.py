from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import ARS, TRPO, QRDQN
from .environment import Environment
from .utils import latest_model
from tqdm import tqdm
import csv
import os

FILE = 'models_data.csv'
final_teste_data = 'final_test_data.csv'


def test_model(algorithm, algo_name, n_iter):
    env = Environment(max_tries_before_change=1)
    model_path = latest_model(algo_name)
    print(f"\nTesting {model_path}\n")
    model = algorithm.load(model_path, env=env)

    for _ in tqdm(range(n_iter), desc=f"Testing {algo_name}"):
        obs, _ = env.reset()
        goal_position = (round(env.goal_position[0], 4), round(env.goal_position[1], 4))
        reached_goal = False
        time = 0
        final_position = (0, 0)

        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, win, info = env.step(action.item())
            if done:
                reached_goal = win
                time += info["time"]
                final_position = (round(info["gps_readings"][0], 4), round(info["gps_readings"][1], 4))

        write_to_csv({
            'model_name': algo_name,
            'win': 1 if reached_goal else 0,
            'time': round(time, 4),
            'final_robot_position': final_position,
            'goal_position': goal_position
        },
            # FILE
            final_teste_data
        )


def main(algo_name=None):
    if algo_name in ["PPO", "A2C", "DQN", "QRDQN", "ARS", "TRPO"]:
        while True:
            try:
                iterations = int(input("Enter the number of iterations: "))
                if iterations > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        test_model(eval(algo_name), algo_name, iterations)

    else:
        print("\nNo valid model selected. Please select a valid model name.\n")


def write_to_csv(result, file):
    file_exists = os.path.exists(file)
    with open(file, mode='a', newline='') as file:
        fieldnames = ['model_name', 'win', 'time', 'final_robot_position', 'goal_position']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
