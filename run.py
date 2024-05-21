from package import testing, training


BOARD = "door"
# BOARD = "circle"

def choose_model():
    print("Please select an algorithm to train:")
    print("1. PPO")
    print("2. A2C")
    print("3. DQN")
    print("4. QRDQN")
    print("5. ARS")
    print("6. TRPO")

    choice = input("Enter the algorithm name: ")
    if choice == "1":
        return "PPO"
    elif choice == "2":
        return "A2C"
    elif choice == "3":
        return "DQN"
    elif choice == "4":
        return "QRDQN"
    elif choice == "5":
        return "ARS"
    elif choice == "6":
        return "TRPO"
    else:
        raise ValueError("No model selected.")


if __name__ == '__main__':
    print("Welcome to the our training and testing environment.")
    print("Please select an option:")
    print("1. Train a model")
    print("2. Test a model")
    option = input("Enter your choice: ")

    if option == "1":
        training(choose_model(),BOARD)
    elif option == "2":
        testing(choose_model(),BOARD)
    else:
        raise ValueError("No option selected.")
