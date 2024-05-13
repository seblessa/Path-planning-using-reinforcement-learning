# from package import

from package import testing, training


if __name__ == '__main__':
    print("Welcome to the our training and testing environment.")
    print("Please select an option:")
    print("1. Train a model")
    print("2. Test a model")
    option = input("Enter your choice: ")

    if option == "1":
        # PPO, SAC, DDPG, DQN
        print("Please select an algorithm to train:")
        print("1. PPO")
        print("2. SAC")
        print("3. DDPG")
        print("4. DQN")

        choice = input("Enter the algorithm name: ")
        if choice == "1":
            training("PPO")
        elif choice == "2":
            training("SAC")
        elif choice == "3":
            training("DDPG")
        elif choice == "4":
            training("DQN")
        else:
            print("No model selected.")

    else:  # option == "2"
        print("Please select an algorithm to test:")
        print("1. PPO")
        print("2. SAC")
        print("3. DDPG")
        print("4. DQN")

        choice = input("Enter the algorithm name: ")
        if choice == "1":
            testing("PPO")
        elif choice == "2":
            testing("SAC")
        elif choice == "3":
            testing("DDPG")
        elif choice == "4":
            testing("DQN")
        else:
            print("No model selected.")