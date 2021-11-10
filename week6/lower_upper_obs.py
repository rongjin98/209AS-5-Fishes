import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from gridworld_setup import GridWorld
from visualizer import draw_square

def lower_bound_obs(stateSpace, blockSpace, target):
    '''
    Always Floor
    '''
    obs_1 = []
    for state_ in stateSpace:
        if_block = (blockSpace == state_).all(1).any()
        if if_block == True:
            obs_ = 9.9 #Trivial Value, blockSpace will never be included
        else:
            obs_ = calObs(state_,target)
            obs_ = np.floor(obs_)
        obs_1.append(obs_)
    return np.array(obs_1)

def upper_bound_obs(stateSpace, blockSpace, target):
    '''
    Always Ceil
    '''
    obs_2 = []
    for state_ in stateSpace:
        if_block = (blockSpace == state_).all(1).any()
        if if_block == True:
            obs_ = 9.9 #Trivial Value, blockSpace will never be included
        else:
            obs_ = calObs(state_,target)
            obs_ = np.ceil(obs_)
        obs_2.append(obs_)
    return np.array(obs_2)

def actual_obs(stateSpace, target):
    '''
    Always Floor
    '''
    obs_3 = []
    for state_ in stateSpace:
        obs_ = calObs(state_,target)
        obs_ = np.round(obs_,3)
        obs_3.append(obs_)
    return np.array(obs_3)

def floor_ceil_possibility(actual_obs):
    floor = []
    ceil = []
    for obs_ in actual_obs:
        pr_floor = np.ceil(obs_)-obs_
        pr_ceil = 1 - pr_floor
        pr_floor = np.round(pr_floor,3)
        pr_ceil = np.round(pr_ceil,3)
        floor.append(pr_floor)
        ceil.append(pr_ceil)
    return floor, ceil

def obser_pr_map(stateSpace, blockSpace, target, obs):
    lower_bound_obs_map = lower_bound_obs(stateSpace, blockSpace, target)
    upper_bound_obs_map = upper_bound_obs(stateSpace, blockSpace, target)
    pr_lower_bound, pr_upper_bound = floor_ceil_possibility(actual_obs(stateSpace, target))

    pr_obs_state = np.zeros(len(stateSpace)) #25*1
    for i in range(len(lower_bound_obs_map)):
        if obs == lower_bound_obs_map[i]:
            pr_obs_state[i] = pr_lower_bound[i]

    for j in range(len(upper_bound_obs_map)):
        if obs == upper_bound_obs_map[j]:
            pr_obs_state[j] = pr_upper_bound[j]
    return pr_obs_state
        
    



def calObs(position, target):
    """
    Calculate observation
    @
    @param position - an array
    return np.float
    """
    RS, RD = target
    distance2RS = np.linalg.norm(position - RS)
    distance2RD = np.linalg.norm(position - RD)
    if distance2RS == 0 or distance2RD == 0:
        h = 0
    else:
        h = 2 / (1 / distance2RS + 1 / distance2RD)
    
    #Now random flooring and ceiling
    return h

if __name__ == "__main__":
    grid = GridWorld([0,2])
    stateSpace = grid.stateSpace
    target = grid.target
    block = grid.blockSpace
    pr_ob_map = obser_pr_map(stateSpace,block,target,2)
    draw_square(pr_ob_map,5)
    # print("==============Lower Bound===============")
    # draw_square(lower_bound_obs(stateSpace,block,target),5)
    # print("==============Upper Bound===============")
    # draw_square(upper_bound_obs(stateSpace,block,target),5)
    # print("==============Actual Obs=================")
    # actual_obsv = actual_obs(stateSpace,target)
    # draw_square(actual_obsv,5)
    # print("==============Floor Prob===============")
    # floor,ceil = floor_ceil_possibility(actual_obsv)
    # draw_square(floor,5)
    # print("==============Ceil Prob===============")
    # draw_square(ceil,5)

