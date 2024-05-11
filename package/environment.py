import gymnasium
from gymnasium import spaces
import math
import numpy as np
from controller import Robot, Lidar, GPS, Supervisor
from package import cmd_vel, move_forward, rotate

from stable_baselines3 import PPO


MOVE_FORWARD = 0
# MOVE_BACKWARDS = 3
ROTATE_LEFT = 1
ROTATE_RIGHT = 2


class Environment(gymnasium.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=math.inf, shape=(100,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.robot: Supervisor = Supervisor()
        self.robot_node = self.robot.getFromDef("robot")

        self.translation_node = self.robot_node.getField("translation")

        self.timestep: int = int(self.robot.getBasicTimeStep())

        self.lidar: Lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.initial_position = (0, 0)
        self.goal_position = (1.50, 1.70)
        self.min_distance = 0.3
        self.lose_distance = 0.1

        self.num_timesteps = 0
        self.max_timesteps = 2000

    def step(self, action):

        if action == 0:
            cmd_vel(self.robot, 0.2, 0)
            self.robot.step(self.timestep)
        elif action == 3:
            cmd_vel(self.robot, -0.1, 0)
            self.robot.step(self.timestep)
        elif action == 1:
            cmd_vel(self.robot, 0, 0.1)
            self.robot.step(self.timestep)
        elif action == 2:
            cmd_vel(self.robot, 0, -0.1)
            self.robot.step(self.timestep)

        self.num_timesteps += 1

        actual_location = self.gps_info()
        lidar_data = self.get_obs()

        reward, terminated = self.calculate_reward(actual_location, lidar_data)
        return np.array(lidar_data, dtype=np.float32), reward, terminated, False, {}

    def get_obs(self):
        lidar_data = self.lidar.getRangeImage()
        lidar_data = np.array(lidar_data)
        lidar_data = np.reshape(lidar_data, (100,))
        return lidar_data

    def gps_info(self):
        gps_readings = self.gps.getValues()
        return (gps_readings[0], gps_readings[1])

    def calculate_reward(self, gps_readings, lidar_data):
        if self.reached_goal(gps_readings):
            return 100, True

        elif self.close_to_obstacle(lidar_data) or self.num_timesteps > self.max_timesteps:
            return -100, True

        else:
            cont = 0
            reward = self.calculate_distance(gps_readings)
            for i in range(len(lidar_data)):
                if lidar_data[i] < self.min_distance:
                    cont += 1
                if math.isinf(lidar_data[i]):
                    lidar_data[i] = 10000
            reward += -0.1 * cont
            reward = round(reward, 2)
            return reward, False

    def calculate_distance(self, gps_readings):
        return round(math.sqrt((gps_readings[0] - self.goal_position[0]) ** 2 + (gps_readings[1] - self.goal_position[1]) **2), 2)

    def reached_goal(self, actual_location):
        if abs(actual_location[0] - self.goal_position[0]) < 0.1 and abs(actual_location[1] - self.goal_position[1]) < 0.1:
            return True
        return False

    def close_to_obstacle(self, lidar_data):
        for i in range(len(lidar_data)):
            if lidar_data[i] < self.lose_distance:
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.simulationReset()
        cmd_vel(self.robot, 0, 0)
        self.num_timesteps = 0

        return 0, {}

'''
if __name__ == '__main__':
    env = Environment()
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, bol, info = env.step(action)
        print(obs)
'''

env = Environment()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs")

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"models/PPO/{TIMESTEPS*iters}")

