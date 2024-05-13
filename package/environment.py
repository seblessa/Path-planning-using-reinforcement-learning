import gymnasium
from gymnasium import spaces
import math
import time
import numpy as np
from controller import Robot, Lidar, GPS, Supervisor
from package import cmd_vel, move_forward, rotate

MOVE_FORWARD = 0
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

        self.battery = self.robot.batterySensorEnable(self.timestep)

        self.lidar: Lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.time_start = 0

        self.last_position = self.gps_info()
        self.last_distance = None

        self.initial_position = (0, 0)
        self.goal_position = (1.50, 1.70)
        self.goal_distance = 0.1

        self.min_safe_distance = 0.3
        self.lose_distance = 0.1

        self.num_timesteps = 0
        self.max_timesteps = 5000
        self.last_action = -1

    def step(self, action):
        if self.last_action == -1:
            self.time_start = time.time()

        if action == 0:
            cmd_vel(self.robot, 0.1, 0)
            self.robot.step(self.timestep)
        elif action == 1:
            cmd_vel(self.robot, 0.1, 0.1)
            self.robot.step(self.timestep)
        elif action == 2:
            cmd_vel(self.robot, 0.1, -0.1)
            self.robot.step(self.timestep)

        self.last_action = action

        self.num_timesteps += 1

        actual_location = self.gps_info()
        lidar_data = self.get_obs()

        reward, terminated = self.calculate_reward(actual_location, lidar_data)
        self.last_position = actual_location
        return np.array(lidar_data, dtype=np.float32), reward, terminated, self.reached_goal(actual_location), {
            "gps_readings": actual_location, "battery": self.robot.batterySensorGetValue(),
            "time": time.time() - self.time_start}

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
            return 10000, True

        elif self.close_to_obstacle(lidar_data) or self.num_timesteps > self.max_timesteps:
            return -10000, True

        else:
            cont = 0
            reward = 4 - self.calculate_distance(gps_readings)
            reward += self.calculate_direction_reward(gps_readings)
            for i in range(len(lidar_data)):
                if lidar_data[i] < self.min_safe_distance:
                    cont += 1
                if math.isinf(lidar_data[i]):
                    lidar_data[i] = 10000
            reward += -0.1 * cont
            reward = round(reward, 2)
            return reward, False

    def calculate_distance(self, gps_readings):
        return round(
            math.sqrt((gps_readings[0] - self.goal_position[0]) ** 2 + (gps_readings[1] - self.goal_position[1]) ** 2),
            2)

    def calculate_direction_reward(self, gps_readings):
        new_closest_point, new_distance = self.point_on_line_closest_to_goal(line_point1=self.last_position,
                                                                             line_point2=gps_readings,
                                                                             given_point=self.goal_position)
        if self.last_distance is None:
            last_closest_point, last_distance = self.point_on_line_closest_to_goal(given_point=self.goal_position, m=1,
                                                                                   b=0)
            self.last_distance = last_distance
        if new_distance < self.last_distance:
            return 1
        else:
            return 0

    def point_on_line_closest_to_goal(self, line_point1=None, line_point2=None, given_point=None, m=None, b=None):

        # Calculate the equation of the line (y = mx + b)
        if m is None:
            m = (line_point2[1] - line_point1[1]) / (line_point2[0] - line_point1[0])

        if b is None:
            b = line_point1[1] - m * line_point1[0]

        # Calculate the perpendicular distance from the given point to the line
        distance = np.abs(m * given_point[0] - given_point[1] + b) / np.sqrt(m ** 2 + 1)

        # Calculate the x-coordinate of the point on the line closest to the given point
        x_closest = (given_point[0] + m * given_point[1] - m * b) / (m ** 2 + 1)

        # Calculate the y-coordinate of the point on the line closest to the given point
        y_closest = m * x_closest + b

        return (x_closest, y_closest), distance

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
        self.robot.simulationReset()
        cmd_vel(self.robot, 0, 0)
        self.num_timesteps = 0
        self.last_action = -1
        obs = self.get_obs()
        for i in range(len(obs)):
            if math.isinf(obs[i]):
                obs[i] = 10000
        return obs, {}
