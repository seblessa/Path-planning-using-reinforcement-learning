from package import testing, training, choose_model


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
