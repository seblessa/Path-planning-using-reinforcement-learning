from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import ARS, TRPO, QRDQN
from .environment import Environment
from .metrics import print_metrics
from .utils import latest_model


def test_model(algorithm, algo_name):
    env = Environment()
    model_path = latest_model(algo_name)

    print(f"\nTesting {model_path}\n")

    model = algorithm.load(model_path, env=env)

    num_wins = 0
    episodes = 5
    goal_position = env.goal_position
    width_object = env.goal_distance * 2
    total_time = 0
    position = (0, 0)
    list_position = list(position)

    for ep in range(episodes):
        print(f"Episode {ep + 1}.")
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, win, info = env.step(action.item())
            # print(info["gps_readings"][0])
            if done:
                print(f"Episode {ep + 1} finished.")
            if win:
                num_wins += 1
                total_time += info["time"]

                list_position[0] += info["gps_readings"][0]
                list_position[1] += info["gps_readings"][1]
                position = tuple(list_position)
                print("Goal Reached")

    metrics_info = {"num_episodes": episodes, "num_successful_trials": num_wins, "time_to_reach_goal": total_time,
                    "final_position": position,
                    "goal_position": goal_position,
                    "width_object": width_object}

    print_metrics(metrics_info)


def main(algo_name=None):
    if algo_name == "PPO":
        test_model(PPO, algo_name)
    elif algo_name == "A2C":
        test_model(A2C, algo_name)
    elif algo_name == "DQN":
        test_model(DQN, algo_name)
    elif algo_name == "QRDQN":
        test_model(QRDQN, algo_name)
    elif algo_name == "ARS":
        test_model(ARS, algo_name)
    elif algo_name == "TRPO":
        test_model(TRPO, algo_name)
    else:
        print("\nNo model selected.\n\n\n")
        main(algo_name)
