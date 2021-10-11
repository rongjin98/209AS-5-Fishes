from gridworld_setup import GridWorld
import numpy as np

grid = GridWorld() #tbd

'''
pe, Rd, Rs, Rw, discount, Time Horizon
'''

def reward_function():
    reward_map = []
    for state_ in grid.stateSpace:
        if_wall = (grid.wall == state_).all(1).any()
        if_block = (grid.blockSpace == state_).all(1).any()
        if_target = (grid.target == state_).all(1).any()
        if if_wall == True:
            reward_map.append(-1)
        elif if_target == True:
            if np.array_equal(state_,grid.target[0]):
                reward_map.append(10)
            else:
                reward_map.append(1)
        elif if_block == True:
            reward_map.append(-99) #not important, wont consider block in calculation
        else:
            reward_map.append(0)
    return reward_map


# def value_function_iteration(reward_map, transit_prob):

#     return NotImplementedError

# def policy_function_iteration(reward_map, policy):

#     return NotImplementedError


if __name__ == "__main__":
    reward = reward_function()
    for i in range(grid.gridSize):
        sth = []
        for j in range(grid.gridSize):
            sth.append(reward[5*i+j])
        print(sth)