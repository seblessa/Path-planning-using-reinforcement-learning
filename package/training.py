from stable_baselines3 import PPO, SAC, DDPG, DQN
from environment import Environment
from utils import latest_model
import os

TIMESTEPS = 10 ** 4
models_dir = "models"
logdir = "logs"
policy = "MlpPolicy"


def train_model(algo, algo_name, policy):
    env = Environment()

    if os.path.exists(f"{models_dir}/{algo_name}"):
        if os.listdir(f"{models_dir}/{algo_name}"):

            model_path = latest_model(algo_name)
            model = algo.load(model_path, env=env)
            iters = int(int(model_path.split("/")[2].split(".")[0]) / TIMESTEPS)
        else:
            model = algo(policy, env, verbose=1,
                         tensorboard_log=logdir)
            iters = 0
    else:
        os.makedirs(f"{models_dir}/{algo_name}")
        model = algo(policy, env, verbose=1,
                     tensorboard_log=logdir)
        iters = 0

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algo_name)
        model.save(f"{models_dir}/{algo_name}/{TIMESTEPS * iters}")
        if iters * TIMESTEPS >= 60 * 10 ** 6:
            break


def main(algo_name=None):
    try:
        if algo_name is None:
            raise ValueError("No algorithm given. Please specify which model to train.")

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        if algo_name == "PPO":
            train_model(PPO, algo_name, policy)
        elif algo_name == "SAC":
            train_model(SAC, algo_name, policy)
        elif algo_name == "DDPG":
            train_model(DDPG, algo_name, policy)
        elif algo_name == "DQN":
            train_model(DQN, algo_name, policy)

        else:
            raise ValueError("Invalid argument. Please specify a valid algorithm.")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main("PPO")
