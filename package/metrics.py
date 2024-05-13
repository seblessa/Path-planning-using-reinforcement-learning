import math


def success_rate(num_episodes, num_successful_trials):
    return num_successful_trials / num_episodes


def time_to_reach_goal(num_episodes, time_to_reach_goal):
    return time_to_reach_goal / num_episodes


def distance_to_goal(gps_readings, goal_position):
    width_of_object = 0.2
    distance_to_goal = math.sqrt((gps_readings[0] - goal_position[0]) ** 2 + (gps_readings[1] - goal_position[1]) ** 2)
    return width_of_object / distance_to_goal

def energy_consumption(num_episodes, energy_consumption):
    return energy_consumption / num_episodes

# TODO: Improve metrics and add more if needed
