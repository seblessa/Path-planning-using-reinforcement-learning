import gymnasium
from gymnasium import spaces
import math
import time
import numpy as np
from .map_script import generate_map
from controller import Robot, Lidar, GPS, Supervisor
from package import cmd_vel, move_forward, rotate

MOVE_FORWARD = 0
ROTATE_LEFT = 1
ROTATE_RIGHT = 2


class Environment(gymnasium.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=math.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        generate_map()

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

        self.time_start = 0

        self.random_position = True
        self.max_tries = 10
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
        #lidar_data = np.reshape(lidar_data, (20,))
        return lidar_data

    def gps_info(self):
        gps_readings = self.gps.getValues()
        return gps_readings[0], gps_readings[1]

    def calculate_reward(self, gps_readings, lidar_data):
        '''
        if self.reached_goal(gps_readings):
            return 1000, True

        elif self.close_to_obstacle(lidar_data) or self.num_timesteps > self.max_timesteps:
            return -200, True
        else:
            # reward = 4 - self.calculate_distance(gps_readings)
            reward += self.calculate_direction_reward(gps_readings)
            for i in range(len(lidar_data)):
                if lidar_data[i] < self.min_safe_distance:
                    cont += 1
                if math.isinf(lidar_data[i]):
                    lidar_data[i] = 3
            reward += -0.5 * cont
            reward = round(reward, 2)
            return reward, False
        '''

        done = False
        distance = self.calculate_distance(gps_readings)
        normalized_distance = distance / 2
        normalized_distance *= 100
        reward = 0

        if normalized_distance < 42:
            if normalized_distance < 10:
                growth_factor = 5
                A = 2.5
            elif normalized_distance < 25:
                growth_factor = 4
                A = 1.5
            elif normalized_distance < 37:
                growth_factor = 2.5
                A = 1.2
            else:
                growth_factor = 1.2
                A = 0.9
            reward += A * (1 - np.exp(-growth_factor * (1 / normalized_distance)))

        else:
            reward += -normalized_distance / 100

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

        return reward, done

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
            return -1

    @staticmethod
    def point_on_line_closest_to_goal(line_point1=None, line_point2=None, given_point=None, m=None, b=None):

        # Calculate the equation of the line (y = mx + b)
        if m is None:
            if line_point2[0] == line_point1[0]:
                m = 1000
            else:
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
        self.counter += 1

        # After TIMEOUT, randomize the initial and final positions
        if self.random_position:
            if self.counter >= self.max_tries:
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
