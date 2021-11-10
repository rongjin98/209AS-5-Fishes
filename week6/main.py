import numpy as np
import random
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from gridworld_setup import GridWorld
from visualizer import draw_square
from simulator import Simulator
from bayes import Bayes


if __name__ == "__main__":
    """
    Initialization
    """
    # action_sequence = np.array([0,1,2,3,4])
    initial_state = [0,2]
    grid = GridWorld(initial_state)
    simulate = Simulator(grid)

    inital_index = Bayes.position_to_index(initial_state,grid.gridSize)
    initial_belief = np.zeros(len(grid.stateSpace))  #bel_0_+
    initial_belief[inital_index] = 1

    """
    Baynes Filter Localization
    """
    while (grid.target == grid.current_position).all(1).any() == False:
    # for action_ in action_sequence:
        action_ = random.randint(0,4)
        next_state = simulate.update_state(action_)
        grid = GridWorld(next_state)
        simulate = Simulator(grid)
        bayes = Bayes(grid,initial_belief,action_)
        initial_belief = bayes.belief_posterior
        estimate_state = np.unravel_index(initial_belief.argmax(), (5,5))
        print("The estimated state is: ", estimate_state)
        draw_square(initial_belief,grid.gridSize)
    
    estimate_state = np.unravel_index(initial_belief.argmax(), (5,5))
    print("The final estimated state is: ", estimate_state)

