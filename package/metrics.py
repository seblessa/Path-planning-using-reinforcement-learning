import math


def success_rate(num_episodes, num_successful_trials):
    return num_successful_trials / num_episodes


def time_to_reach_goal(num_episodes, total_time):
    return round(total_time / num_episodes, 2)


def distance_to_goal(num_episodes, gps_readings, goal_position):
    if gps_readings == (0, 0):
        return 0
    mean_readings_x = gps_readings[0] / num_episodes
    mean_readings_y = gps_readings[1] / num_episodes
    goal_distance = math.sqrt((mean_readings_x - goal_position[0]) ** 2 + (mean_readings_y - goal_position[1]) ** 2)
    return abs(round(goal_distance, 2))


def print_metrics(info):
    num_episodes = info["num_episodes"]
    num_successful_trials = info["num_successful_trials"]
    time = info["time_to_reach_goal"]
    final_position = info["final_position"]
    goal_position = info["goal_position"]
    # width_object = info["width_object"]

    successrate = success_rate(num_episodes, num_successful_trials)
    time = time_to_reach_goal(num_episodes, time)
    distance = distance_to_goal(num_episodes,final_position, goal_position)

    if successrate == 0:
        print("\nThe robot was not able to reach the goal\n")
    else:
        print(f"\nThe sucess rate was {successrate * 100}%")
        print(f"The median time to reach the goal was {time} seconds")
        print(f"The average distance of the robot from the goal was {distance} meters")
