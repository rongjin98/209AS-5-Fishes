import numpy as np
from gridworld_setup import GridWorld
import random

grid = GridWorld()

#just a helper function for plotting
def draw_square(array):
    print("--------------------------")
    for i in range(grid.gridSize):
        sth = []
        for j in range(grid.gridSize):
            sth.append(array[5*i+j])
        print(sth)
    print("--------------------------")
    return None

def draw_action(array):
    symbol_array = []
    for i in range(len(array)):
        if array[i] == 0:
            symbol_array.append(u'\u2191')
        elif array[i] == 1:
            symbol_array.append(u'\u2193')
        elif array[i] == 2:
            symbol_array.append(u'\u2190')
        elif array[i] == 3:
            symbol_array.append(u'\u2192')
        elif array[i] == 4:
            symbol_array.append(u'\u2298')
    return symbol_array


def reward_function(Rd, Rs, Rw): 
    reward_map = []
    for state_ in grid.stateSpace:
        if_wall = (grid.wall == state_).all(1).any()
        if_block = (grid.blockSpace == state_).all(1).any()
        if_target = (grid.target == state_).all(1).any()
        if if_wall == True:
            reward_map.append(Rw)
        elif if_target == True:
            if np.array_equal(state_,grid.target[0]):
                reward_map.append(Rs)
            else:
                reward_map.append(Rd)
        elif if_block == True:
            reward_map.append(-99) #trivial value, wont consider block in calculation since transition probability is always 0
        else:
            reward_map.append(0)
    return reward_map

def get_available_actions():
    available_action_each_state = []
    for state_ in grid.stateSpace:
        available_action_at_state = []
        if_block_state = (grid.blockSpace == state_).all(1).any()
        if if_block_state == True:
            available_action_each_state.append([4]) #stay
        else:
            for action_index in range(len(grid.actionSpace)):
                next_state = state_ + grid.actionSpace[action_index]
                if_in_block = (grid.blockSpace == next_state).all(1).any()
                if_in_bound = (grid.stateSpace == next_state).all(1).any()
                if if_in_block == False and if_in_bound == True:
                    available_action_at_state.append(action_index)
            available_action_each_state.append(available_action_at_state)
    return np.array(available_action_each_state)


def setup_random_policy(availble_actions):
    policy_map =[]
    for i in range(len(grid.stateSpace)):
        viable_action_set = availble_actions[i]
        pick_random_action = np.random.choice(viable_action_set)
        policy_map.append(pick_random_action)
    return np.array(policy_map)




def policy_function_iteration(discount, reward, H):
    available_action_set = get_available_actions()
    previous_policy = setup_random_policy(available_action_set)
    initial_policy = setup_random_policy(available_action_set)

    v_evaluate  = np.zeros(len(grid.stateSpace))
    epoch = 0

    while np.array_equal(previous_policy,initial_policy) == False and  epoch < H:
        new_policy = []
        #policy evaluation:
        for state_index in range(len(grid.stateSpace)):
            transit_prob_given_action_state = grid.transition_probability[initial_policy[state_index]][state_index]
            policy_evaluation = 0
            index1 = 0
            for value in transit_prob_given_action_state:
                if value > 0:
                    policy_evaluation += value*((reward[index1])+discount*v_evaluate[index1])
                index1 += 1
            v_evaluate[state_index] = policy_evaluation
        
        #policy refinement:
        for state_index2 in range(len(grid.stateSpace)):
            q_max = -999
            action_max = -999
            for action_index in available_action_set[state_index2]:
                transit_prob_given_action_state2 = grid.transition_probability[action_index][state_index2]

                index2 = 0
                q_temp = 0
                
                for value in transit_prob_given_action_state2:
                    if value > 0:
                        q_temp += value*((reward[index2])+discount*v_evaluate[index2])
                    index2 += 1

                if q_temp > q_max:
                    q_max = q_temp
                    action_max = action_index

            #update policy
            new_policy.append(action_max)
        previous_policy = initial_policy
        initial_policy = new_policy

        epoch +=1
        draw_square(v_evaluate)
        draw_square(draw_action(initial_policy))
        print(epoch)    
    return initial_policy
        

if __name__ == "__main__":
    available_action = get_available_actions()
    reward = reward_function(8,10,-10)
    initial_p = policy_function_iteration(0.8,reward,20)
    draw_square(draw_action(initial_p))

