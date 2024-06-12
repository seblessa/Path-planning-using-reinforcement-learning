# Path-planning-using-reinforcement-learning
Assignment for Introduction to Intelligent Robotics Class, 3º Year,2º Semester, Bachelor in Artificial Intelligence and Data Science

# Summary

In this project our goal is to train various models in the [Stable Baselines3](https://stable-baselines.readthedocs.io/en/master/) library to solve the path planning problem in a simulation using the [Webots Simulator](https://cyberbotics.com/).
The main objective is to train the models to navigate without colliding with obstacles and reach the target in the shortest time possible.
We used the [A2C](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html), [ARS](https://sb3-contrib.readthedocs.io/en/master/modules/ars.html), [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [DQN](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html), [QR-DQN](https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html) and [TRPO](https://stable-baselines.readthedocs.io/en/master/modules/trpo.html) algorithms to solve the problem.

**Authors**:
- [Alexandre Marques](https://github.com/AlexandreMarques27)
- [Sebastião Santos Lessa](https://github.com/seblessa/)
- [Margarida Vila Chã](https://github.com/margaridavc/)


# Versions

The versions of the operating systems used to develop and test this application are:
- macOS Sonoma 14.5
- Windows 11

Python Versions:
- 3.12


# Requirements

To keep everything organized and simple, we will use MiniConda to manage our environments.
To create an environment with the required packages for this project, run the following commands:

```bash
conda create -n robotics python pytorch::pytorch torchvision torchaudio -c pytorch
```
To install the requirements run:
```bash
pip install -r requirements.txt
```

# Usage

There are two usaged modes for this project: training and testing. To start run the following command:

```bash
python3 run.py
```

You will be greeted with the following options:

```
Welcome to the our training and testing environment.
Please select an option:
1. Train a model
2. Test a model
Enter your choice:
```
Then you can choose between training a model or testing a model. With both options you will be asked to choose the model you want to use:
```
Please select an algorithm to train:
1. PPO
2. A2C
3. DQN
4. QRDQN
5. ARS
6. TRPO
Enter the algorithm number:
```

If you choose to train a model then the training will start on Webots. To train faster you can toggle the `Rendering` option in the Webots buttons.



If you choose to test a model, you will be asked how many episodes you want to run. The results of all the testing will be sent to a file called [models_data.csv](models_data.csv).


# Results

After testing 1000 episodes for each model, we were able to analyze the results and compare the performance of each model. The results are shown in the following notebook: [metric_analysis.ipynb](metric_analysis.ipynb)
