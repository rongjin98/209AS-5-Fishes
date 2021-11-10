import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from gridworld_setup import GridWorld
from double_agents_transition_prob import two_agents_world
from basis_function import Basis_Function
import visualizer
import time

# grid = GridWorld([0,2])
# two_agent_grid = two_agents_world(grid,[0,2],[0,3])



def value_iteration_approximation(Horizon, discount, transition_probability, available_action_set, two_agent_grid, basis_function, threshold):
    transit_pr = transition_probability
    stateSpace = two_agent_grid.two_stateSpace
    reward_space = two_agent_grid.reward_map
    actionSpace = two_agent_grid.two_actionSpace
    agents_gridSize = two_agent_grid.agents_gridSize
    gridSize = two_agent_grid.gridSize

    basis_subset = basis_function.basis_subset
    basis_length = len(basis_subset[0])
    theta = np.zeros((1,basis_length)) #make column vector for multiplication

    v_set = np.zeros(len(stateSpace))
    action_decision = np.zeros(len(stateSpace)) #####CHANGE IT TO ALL 4s
    non_block_substate = non_block_state_index(reward_space)
    v_subset = np.zeros(len(non_block_substate))

    prev_diff = -9999
    diff = 0

    for i in range(Horizon):
        '''
        Step1 - Bellman Backups
        '''    
        #Now we pick a subset of statespace, state_index is no longer available, we need a refined one
        count = 0
        for sub_index in non_block_substate:
            state_ = stateSpace[sub_index]
            agent1_cs = state_[0]
            agent2_cs = state_[1]
        
            v_bar = -9999
            max_action = -9999
            for action_index in available_action_set[sub_index]:
                transit_pr_at_state = transit_pr[action_index][agent1_cs][agent2_cs]

                #next_state_index = successor_state_index(agent1_cs, agent2_cs, actionSpace[action_index], agents_gridSize, gridSize)
                v_temp = 0
                index_ = 0
                for value in transit_pr_at_state:
                    if value > 0:
                        map_index = np.argwhere(non_block_substate == index_)
                        #basis = np.reshape(basis_set[next_state_index],(basis_length,1))
                        basis = np.reshape(basis_subset[map_index],(basis_length,1))
                        v_hat = np.matmul(theta,basis).flatten()[0]
                        v_temp += value*(reward_space[index_]+discount*v_hat)
                    index_ += 1
                
                if v_temp > v_bar:
                    v_bar = v_temp
                    max_action = action_index
            v_set[sub_index] = v_bar
            v_subset[count] = v_bar
            action_decision[sub_index] = max_action
            count += 1
        
        '''
        Step2 - Linear Regression: Update theta
        '''
        #visualizer.draw_square(v_set[25:51],5)
        theta = linear_regression(v_subset, theta, basis_subset)
        prev_diff = diff
        diff = check_convergence(theta, basis_subset, v_subset)
        diff_diff = abs(prev_diff- diff)
        print("At iteration:", i, ", the difference is: ", diff_diff)
        if diff_diff < threshold: 
            print("Value Iteration executed at ", i)
            return v_set, action_decision, theta
        
    return v_set, action_decision, theta

def successor_state_index(agent1_cs, agent2_cs, action_index, agents_gridSize, gridSize):
    agent1_position = two_agents_world.index_to_position(agent1_cs,gridSize)
    agent2_position = two_agents_world.index_to_position(agent2_cs, gridSize)

    action_1 = two_agents_world.index_to_action(action_index[0])
    action_2 = two_agents_world.index_to_action(action_index[1])

    agent1_new = agent1_position + action_1
    agent2_new = agent2_position + action_2

    agent1_index = two_agents_world.position_to_index(agent1_new, gridSize)
    agent2_index = two_agents_world.position_to_index(agent2_new, gridSize)

    index_in_stateSpace = agent1_index*agents_gridSize + agent2_index
    return index_in_stateSpace


def check_convergence(theta, basis_set, v_set):
    v_estimate = np.matmul(basis_set, np.transpose(theta)).flatten()
    diff = np.linalg.norm(v_estimate - v_set)
    return diff

def linear_regression(v_set, theta, basis_set):
    y = np.reshape(v_set,(len(v_set),1))
    x = basis_set
    pseudo_inverse = np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x))
    theta = np.matmul(pseudo_inverse,y)
    theta = np.reshape(theta, (1, len(theta)))
    return theta

def block_state_index(reward_map):
    set = []
    index_count = 0
    for i in reward_map:
        if i == -99:
            set.append(index_count)
        index_count += 1
    return np.array(set)

def non_block_state_index(reward_map):
    block_index = block_state_index(reward_map)
    state_index = np.arange(625)
    non_block_state = np.delete(state_index,block_index)
    return non_block_state


#Testing
if __name__ == "__main__":
    grid = GridWorld([0,2])
    two_agent_grid = two_agents_world(grid,[0,2],[4,4])
    transition_prob = np.load('saved_data/transition_probability.npy')
    available_action_set = np.load('saved_data/Available_action_set.npy', allow_pickle = True)
    subspace = non_block_state_index(two_agent_grid.reward_map)
    basis_function = Basis_Function(grid.gridSize,grid.blockSpace, grid.target, two_agent_grid.two_stateSpace, subspace)
    value_set, action_set, theta = value_iteration_approximation(50, 0.8, transition_prob, available_action_set, two_agent_grid, basis_function, 0.001)
    np.save('saved_data/Aprox_Value',value_set)
    np.save('saved_data/Aprox_Value_Policy',action_set)
    np.save('saved_data/theta', theta)
