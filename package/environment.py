import gymnasium
from gymnasium import spaces
import math
import time
import numpy as np
from controller import Lidar, Supervisor
from package import cmd_vel

MOVE_FORWARD = 0
ROTATE_LEFT = 1
ROTATE_RIGHT = 2


class Environment(gymnasium.Env):
    def __init__(self, max_tries_before_change=10):
        self.observation_space = spaces.Box(low=0, high=math.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.robot: Supervisor = Supervisor()
        self.robot_node = self.robot.getFromDef("robot")
        self.translation_node = self.robot_node.getField("translation")

        self.start_node = self.robot.getFromDef("start")

        self.goal_node = self.robot.getFromDef("goal")

        self.timestep: int = int(self.robot.getBasicTimeStep())

        self.lidar: Lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        # self.camera = self.robot.getDevice('camera')
        # self.camera.enable(self.timestep)

        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        self.time_start = 0

        self.random_position = True
        self.max_tries_before_change = max_tries_before_change
        self.counter = 0

        self.last_position = self.gps_info()
        self.last_distance = None

        self.initial_position = (0.05, 0.05)
        self.goal_position = (1.50, 1.70)
        self.goal_distance = 0.1

        self.min_safe_distance = 0.5
        self.lose_distance = 0.1

        self.num_timesteps = 0
        self.max_timesteps = 5000
        self.last_action = -1

        self.last_reward = 0

        self.min_dist = 0.05
        self.max_dist = 1.95

    def step(self, action):
        if self.last_action == -1:
            self.time_start = time.time()

        if action == 0:
            cmd_vel(self.robot, 0.1, 0)
            self.robot.step(self.timestep)
        elif action == 1:
            cmd_vel(self.robot, 0.05, 0.5)
            self.robot.step(self.timestep)
        elif action == 2:
            cmd_vel(self.robot, 0.05, -0.5)
            self.robot.step(self.timestep)

        self.last_action = action

        self.num_timesteps += 1

        actual_location = self.gps_info()
        lidar_data = self.get_obs()

        reward, terminated = self.calculate_reward(actual_location, lidar_data)
        atual_reward = reward - self.last_reward
        self.last_reward = reward
        self.last_position = actual_location
        # print(atual_reward)
        return np.array(lidar_data, dtype=np.float32), atual_reward, terminated, self.reached_goal(actual_location), {
            "gps_readings": actual_location, "time": time.time() - self.time_start}

    def get_obs(self):
        lidar_data = self.lidar.getRangeImage()
        lidar_data = np.array(lidar_data)
        return lidar_data

    def gps_info(self):
        gps_readings = self.gps.getValues()
        return gps_readings[0], gps_readings[1]

    @staticmethod
    def distance_reward(distance):
        normalized_distance = distance / 2
        normalized_distance *= 100

        if normalized_distance < 42:
            if normalized_distance < 5:
                rate = 8
                multiplier = 4
            elif normalized_distance < 10:
                rate = 5
                multiplier = 2.5
            elif normalized_distance < 25:
                rate = 4
                multiplier = 1.5
            elif normalized_distance < 37:
                rate = 2.5
                multiplier = 1.2
            else:
                rate = 1.2
                multiplier = 0.9

            return multiplier * (1 - np.exp(-rate * (1 / normalized_distance)))

        else:
            return -normalized_distance / 100

    def calculate_distance(self, gps_readings):
        return round(
            math.sqrt((gps_readings[0] - self.goal_position[0]) ** 2 + (gps_readings[1] - self.goal_position[1]) ** 2),
            2)

    def calculate_reward(self, gps_readings, lidar_data):
        done = False
        distance = self.calculate_distance(gps_readings)
        reward = self.distance_reward(distance)
        # print(f"DISTANCE REWARD: {reward}")

        direction_reward = self.direction_reward(gps_readings, self.compass.getValues())
        reward += direction_reward
        # print(f"DIRECTION REWARD: {direction_reward}\n")

        for i in range(len(lidar_data)):
            if math.isinf(lidar_data[i]):
                lidar_data[i] = 3

        if self.reached_goal(gps_readings):
            reward += 25
            done = True
        elif self.close_to_obstacle(lidar_data) or self.num_timesteps > self.max_timesteps:
            reward -= 5
            done = True
        elif np.any(lidar_data[lidar_data > self.min_safe_distance]):
            reward -= 0.001

        # print(f"REWARD: {reward}\n\n")
        return reward, done

    def direction_reward(self, current_position, compass_vector):
        # Calculate the goal vector from current_position to goal_position
        goal_vector_x = self.goal_position[0] - current_position[0]
        goal_vector_y = self.goal_position[1] - current_position[1]

        # Calculate the angle of the goal vector
        goal_angle = math.atan2(goal_vector_y, goal_vector_x)

        # Calculate the angle of the compass vector
        current_angle = (math.pi / 2) - math.atan2(compass_vector[1], compass_vector[0])

        # Calculate the difference between the angles
        angle_diff = goal_angle - current_angle

        return math.cos(angle_diff)

    def reached_goal(self, actual_location):
        if abs(actual_location[0] - self.goal_position[0]) < self.goal_distance and abs(
                actual_location[1] - self.goal_position[1]) < self.goal_distance:
            return True
        return False

    def close_to_obstacle(self, lidar_data):
        for i in range(len(lidar_data)):
            if lidar_data[i] < self.lose_distance:
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter += 1

        # After TIMEOUT, randomize the initial and final positions
        if self.random_position:
            if self.counter >= self.max_tries_before_change:
                self.counter = 0
                self.initial_position = (
                    np.random.uniform(self.min_dist, self.max_dist), np.random.uniform(self.min_dist, self.max_dist))
                self.goal_position = (
                    np.random.uniform(self.min_dist, self.max_dist), np.random.uniform(self.min_dist, self.max_dist))
            self.translation_node.setSFVec3f([self.initial_position[0], self.initial_position[1], 0.0])
            self.goal_node.getField("translation").setSFVec3f([self.goal_position[0], self.goal_position[1], -0.049])
            self.start_node.getField("translation").setSFVec3f(
                [self.initial_position[0], self.initial_position[1], -0.049])
            self.robot_node.resetPhysics()
            self.robot.step()
        else:
            self.robot.simulationReset()

        cmd_vel(self.robot, 0, 0)
        self.last_reward = 0
        self.num_timesteps = 0
        self.last_action = -1
        self.time_start = time.time()
        obs = self.get_obs()
        for i in range(len(obs)):
            if math.isinf(obs[i]):
                obs[i] = 10000
        return obs, {}
