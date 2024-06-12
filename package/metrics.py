import math


def success_rate(num_episodes, num_successful_trials):
    return num_successful_trials / num_episodes


def time_to_reach_goal(num_episodes, total_time):
    return total_time / num_episodes


def distance_to_goal(gps_readings, goal_position):
    if gps_readings == (0, 0):
        return 0
    goal_distance = math.sqrt((gps_readings[0] - goal_position[0]) ** 2 + (gps_readings[1] - goal_position[1]) ** 2)
    return abs(goal_distance)


def print_metrics(info):
    num_episodes = info["num_episodes"]
    num_successful_trials = info["num_successful_trials"]
    time = info["time_to_reach_goal"]
    final_position = info["final_position"]
    goal_position = info["goal_position"]
    # width_object = info["width_object"]

    successrate = success_rate(num_episodes, num_successful_trials)
    time = time_to_reach_goal(num_episodes, time)
    distance = distance_to_goal(final_position, goal_position)

    if successrate == 0:
        print("\nThe robot was not able to reach the goal\n")
    else:
        print(f"\nThe sucess rate was {successrate * 100}%")
        print(f"The median time to reach the goal was {time} seconds")
        print(f"The average distance of the robot from the goal was {distance} meters")
