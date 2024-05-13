# from package import

from package import testing, training


if __name__ == '__main__':
    print("Welcome to the our training and testing environment.")
    print("Please select an option:")
    print("1. Train a model")
    print("2. Test a model")
    option = input("Enter your choice: ")

    if option == "1":
        # PPO, A2C, DQN
        print("Please select an algorithm to train:")
        print("1. PPO")
        print("2. A2C")
        print("3. DQN")

        choice = input("Enter the algorithm name: ")
        if choice == "1":
            training("PPO")
        elif choice == "2":
            training("A2C")
        elif choice == "3":
            training("DQN")
        else:
            print("No model selected.")

    else:  # option == "2"
        print("Please select an algorithm to test:")
        print("1. PPO")
        print("2. A2C")
        print("3. DQN")

        choice = input("Enter the algorithm name: ")
        if choice == "1":
            testing("PPO")
        elif choice == "2":
            testing("A2C")
        elif choice == "3":
            testing("DQN")
        else:
            print("No model selected.")
