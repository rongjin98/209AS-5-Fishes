import numpy as np
from double_agents_function_approximation import non_block_state_index
from gridworld_setup import GridWorld
from double_agents_transition_prob import two_agents_world
from basis_function import Basis_Function
import visualizer

def value_iteration(discount, transition_probability, available_action_set, two_agent_grid, basis_set, theta):
    transit_pr = transition_probability
    stateSpace = two_agent_grid.two_stateSpace
    reward_space = two_agent_grid.reward_map

    v_set = np.zeros(len(stateSpace))
    action_decision = np.full(len(stateSpace), 24) #the last index of two agents action space
    basis_length = len(basis_set[0])

    count = 0
    for state_ in stateSpace:
        agent1_cs = state_[0]
        agent2_cs = state_[1]
        
        v_bar = -9999
        max_action = -9999
        for action_index in available_action_set[count]:
            transit_pr_at_state = transit_pr[action_index][agent1_cs][agent2_cs]
            v_temp = 0
            index_ = 0
            for value in transit_pr_at_state:
                if value > 0:
                    #basis = np.reshape(basis_set[next_state_index],(basis_length,1))
                    basis = np.reshape(basis_set[index_],(basis_length,1))
                    v_hat = np.matmul(theta,basis).flatten()[0]
                    v_temp += value*(reward_space[index_]+discount*v_hat)
                index_ += 1
                
            if v_temp > v_bar:
                v_bar = v_temp
                max_action = action_index

        v_set[count] = v_bar
        action_decision[count] = max_action
        count += 1
    return v_set, action_decision

if __name__ == "__main__":
    grid = GridWorld([0,2])
    two_agent_grid = two_agents_world(grid,[0,2],[4,4])
    transition_prob = np.load('saved_data/transition_probability.npy')
    theta = np.load('saved_data/theta.npy')
    subspace = non_block_state_index(two_agent_grid.reward_map)
    basis_function = Basis_Function(grid.gridSize,grid.blockSpace, grid.target, two_agent_grid.two_stateSpace, subspace)
    full_basis = basis_function.basis_fullset
    available_action_set = np.load('saved_data/Available_action_set.npy', allow_pickle = True)
    value_set, action_set= value_iteration(0.8, transition_prob, available_action_set, two_agent_grid, full_basis, theta)

    np.save('saved_data/value_set',value_set)
    np.save('saved_data/policy_set',action_set)

    #What's next?
    #How to make the value change policy with respect to icecream reward?
    #Adjust the reward when two robots crash
    #Input the initial position of two agents, and map their actions until they reach the icecream



    
