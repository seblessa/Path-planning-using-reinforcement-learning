from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import ARS, TRPO, QRDQN
from .environment import Environment
from .utils import latest_model
import os

TIMESTEPS = 10 ** 4
models_dir = "models"
logdir = "logs"
policy = "MlpPolicy"


def train_model(algo, algo_name, board):
    env = Environment()

    if os.path.exists(f"{models_dir}/{algo_name}/{board}"):
        if os.listdir(f"{models_dir}/{algo_name}/{board}"):
            model_path = latest_model(algo_name,board)
            if model_path.endswith("final.zip"):
                print(f"\nModel {model_path} is a final model. Exiting...\n")
                return
            model = algo.load(model_path, env=env)
            iters = int(int(model_path.split("/")[3].split(".")[0]))
            print(f"\nLoaded model with {iters} iterations\n")
        else:
            model = algo(policy, env, verbose=1,
                         tensorboard_log=f"{logdir}/{board}")
            iters = 0
            print("\nCreating new model with no iterations\n")
    else:
        os.makedirs(f"{models_dir}/{algo_name}/{board}")
        model = algo(policy, env, verbose=1,
                     tensorboard_log=f"{logdir}/{board}")
        iters = 0
        print("\nCreating new model with no iterations\n")

    print(f"Training {models_dir}/{algo_name}/{board}")

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algo_name)
        model.save(f"{models_dir}/{algo_name}/{board}/{TIMESTEPS * iters}")


def main(algo_name=None, board=None):
    if algo_name == "PPO":
        print("Training PPO")
        train_model(PPO, algo_name, board)
    elif algo_name == "A2C":
        print("Training A2C")
        train_model(A2C, algo_name, board)
    elif algo_name == "DQN":
        print("Training DQN")
        train_model(DQN, algo_name, board)
    elif algo_name == "QRDQN":
        print("Training QRDQN")
        train_model(QRDQN, algo_name, board)
    elif algo_name == "ARS":
        print("Training ARS")
        train_model(ARS, algo_name, board)
    elif algo_name == "TRPO":
        print("Training TRPO")
        train_model(TRPO, algo_name, board)


if __name__ == '__main__':
    main("PPO")
