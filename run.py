# from package import

import math
from controller import Robot
from package import cmd_vel, move_forward, rotate


if __name__ == '__main__':
    # Create the Robot instance.
    robot: Robot = Robot()

    # tile length in meters
    d = 0.125 / 2

    while True:
        move_forward(robot, d * 2, 0.1)
        rotate(robot, math.pi / 2, 0.5)

