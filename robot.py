# This is a robot class to create a robot that can take action based on some certain strategy

import random

class Robot:
    def __init__(self) -> None:
        pass

    def makeDecision(self) -> int:
        # this function is used to let robot make a decision about where to move
        # currently just randomly pick one direction

        # NoMove, Up, Down, Right, Left
        #   0     1     2     3     4

        #TODO
        # add more strategy

        return random.randint(0,4)