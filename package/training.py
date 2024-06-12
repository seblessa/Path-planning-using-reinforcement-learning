from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import ARS, TRPO, QRDQN
from .environment import Environment
from .utils import latest_model
import os

TIMESTEPS = 10 ** 4
models_dir = "models"
logdir = "logs"
policy = "MlpPolicy"


def train_model(algo, algo_name):
    env = Environment()

    if os.path.exists(f"{models_dir}/{algo_name}"):
        if os.listdir(f"{models_dir}/{algo_name}"):
            model_path = latest_model(algo_name)
            if model_path.endswith("final.zip"):
                print(f"\nModel {model_path} is a final model. Exiting...\n")
                return
            model = algo.load(model_path, env=env)
            print(model_path)
            iters = int(int(model_path.split("/")[2].split(".")[0]))
            print(f"\nLoaded model with {iters} iterations\n")
        else:
            model = algo(policy, env, verbose=1,
                         tensorboard_log=f"{logdir}")
            iters = 0
            print("\nCreating new model with no iterations\n")
    else:
        os.makedirs(f"{models_dir}/{algo_name}")
        model = algo(policy, env, verbose=1,
                     tensorboard_log=f"{logdir}")
        iters = 0
        print("\nCreating new model with no iterations\n")

    print(f"Training {models_dir}/{algo_name}")

    while True:
        iters += TIMESTEPS
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algo_name)
        model.save(f"{models_dir}/{algo_name}/{iters}")


def main(algo_name=None):
    if algo_name in ["PPO", "A2C", "DQN", "QRDQN", "ARS", "TROPO"]:
        print(f"Training {algo_name}")
        train_model(eval(algo_name), algo_name)
    else:
        print("\nNo valid model selected. Please select a valid model name.\n\n")
