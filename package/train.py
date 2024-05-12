from environment import Environment
from stable_baselines3 import PPO

env = Environment()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs")

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"models/PPO/{TIMESTEPS*iters}")