import math


def success_rate(num_episodes, num_successful_trials):
    return num_successful_trials / num_episodes


def time_to_reach_goal(num_episodes, time_to_reach_goal):
    return time_to_reach_goal / num_episodes


def distance_to_goal(gps_readings, goal_position, width_of_object):
    if gps_readings == (0, 0):
        return 0
    distance_to_goal = math.sqrt((gps_readings[0] - goal_position[0]) ** 2 + (gps_readings[1] - goal_position[1]) ** 2)
    return width_of_object / distance_to_goal


def energy_consumption(num_episodes, energy_consumption):
    return energy_consumption / num_episodes


def print_metrics(info):
    num_episodes = info["num_episodes"]
    num_successful_trials = info["num_successful_trials"]
    time = info["time_to_reach_goal"]
    distance = info["distance_to_goal"]
    goal_position = info["goal_position"]
    width_object = info["width_object"]
    energy = info["energy_consumption"]

    Success_rate = success_rate(num_episodes, num_successful_trials)
    Time_to_reach_goal = time_to_reach_goal(num_episodes, time)
    Distance_to_goal = distance_to_goal(distance, goal_position, width_object)
    Energy_consumption = energy_consumption(num_episodes, energy)

    if Success_rate == 0:
        print("The robot was not able to reach the goal")
    else:
        print(f"The sucess rate was {Success_rate}%")
        print(f"The median time to reach the goal was {Time_to_reach_goal} seconds")
        print(f"The average distance of the robot from the goal was {Distance_to_goal} meters")
        if Energy_consumption == -1:
            print("The robot did not have the battery turned on.")
        else:
            print(f"The mean energy consumption was {Energy_consumption} Jules")
