from package import testing, training
import platform
import sys

if platform.system() == "Darwin":
    webots_path = "/Applications/Webots.app/Contents/lib/controller/python"
    print("macOS")
else:  # Windows
    #webots_path = "C:\Program Files\Webots\lib\controller\python"
    print("Windows")

sys.path.append(webots_path)


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
        print("\nNo model selected.\n\n\n")
        return choose_model()


def main():
    print("Welcome to the our training and testing environment.")
    print("Please select an option:")
    print("1. Train a model")
    print("2. Test a model")
    option = input("Enter your choice: ")

    if option == "1":
        training(choose_model())
    elif option == "2":
        testing(choose_model())
    else:
        print("\nInvalid option.\n\n\n")
        main()


if __name__ == '__main__':
    main()
