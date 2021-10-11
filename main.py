import numpy as np
from gridworld_setup import GridWorld
from simulator import Simulator

if __name__ == "__main__":
    grid = GridWorld()
    simulator = Simulator(grid)

    print(simulator.new_state)
