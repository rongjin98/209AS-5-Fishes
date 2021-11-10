import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from gridworld_setup import GridWorld
from simulator import Simulator
from visualizer import draw_square
from visualizer import draw_action
from value_iteration_method import value_function_iteration
from policy_iteration_method import policy_function_iteration

if __name__ == "__main__":
    grid = GridWorld([0,2])
    simulator = Simulator(grid)
    path = [grid.current_position]
    reward_map = grid.reward_function()

    value_function, value_policy = value_function_iteration(0.8,reward_map,100,0.01,grid)
    policy_function = policy_function_iteration(0.8,reward_map,100,grid)
    print("The value function result is shown below: ")
    draw_square(value_function,grid.gridSize)

    print("The policy result of the value iteration method: ")
    draw_square(draw_action(value_policy),grid.gridSize)

    print("The policy result of the policy iteration method: ")
    draw_square(draw_action(policy_function),grid.gridSize)
    count = 0
    while (grid.target == grid.current_position).all(1).any() == False:
        current_index = grid.get_current_state_index()
        new_state = simulator.update_state(value_policy[current_index])
        grid = GridWorld(new_state)
        simulator = Simulator(grid)
        path.append(grid.current_position)
        count += 1
    path = np.array([path])
    print("It takes " , count , " steps to reach the ice-cream shop")
    print(path)
