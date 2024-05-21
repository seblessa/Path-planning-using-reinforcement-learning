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



def print_metrics(info):
    num_episodes = info["num_episodes"]
    num_successful_trials = info["num_successful_trials"]
    time = info["time_to_reach_goal"]
    distance = info["distance_to_goal"]
    goal_position = info["goal_position"]
    width_object = info["width_object"]

    Success_rate = success_rate(num_episodes, num_successful_trials)
    Time_to_reach_goal = time_to_reach_goal(num_episodes, time)
    Distance_to_goal = distance_to_goal(distance, goal_position, width_object)

    if Success_rate == 0:
        print("\nThe robot was not able to reach the goal\n")
    else:
        print(f"\nThe sucess rate was {Success_rate * 100}%")
        print(f"The median time to reach the goal was {Time_to_reach_goal} seconds")
        print(f"The average distance of the robot from the goal was {Distance_to_goal} meters")
