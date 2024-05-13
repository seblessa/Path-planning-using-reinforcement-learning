from stable_baselines3 import PPO, SAC, DDPG, DQN
from .environment import Environment
from .utils import latest_model
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
    if algo_name == "PPO":
        print("Training PPO")
        train_model(PPO, algo_name, policy)
    elif algo_name == "SAC":
        print("Training SAC")
        train_model(SAC, algo_name, policy)
    elif algo_name == "DDPG":
        print("Training DDPG")
        train_model(DDPG, algo_name, policy)
    elif algo_name == "DQN":
        print("Training DQN")
        train_model(DQN, algo_name, policy)


if __name__ == '__main__':
    main("PPO")
