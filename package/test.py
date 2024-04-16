import math
from controller import Robot
from package import cmd_vel, move_forward, rotate

def run():
    # Create the Robot instance.
    robot: Robot = Robot()

    while True:
        #rotate(robot, math.pi/4, 0.5)
        move_forward(robot, 10, 1)
