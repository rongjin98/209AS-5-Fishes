from gridworld_setup import GridWorld
import numpy as np
import random

grid = GridWorld() #tbd

'''
pe, Rd, Rs, Rw, discount, Time Horizon
'''
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



def value_function_iteration(discount, reward, H, threshold):
    v_previous = np.random.rand(len(grid.stateSpace)) #initial value at time-step = 0
    v_i = np.zeros(len(grid.stateSpace)) #value at time-step = 0x
    available_action_set = get_available_actions()
    for i in range(H):
        value_based_policy = []
        for state_index in range(len(grid.stateSpace)):
            '''
            we need to calculate value function for each state
            '''
            v_max = -999
            action_max = -999
            for action_index in available_action_set[state_index]:
                '''
                loop through the whole actionspace to find the action to maximize value function
                '''
                transit_prob_given_action_state = grid.transition_probability[action_index][state_index]
                index = 0
                v_temp = 0
                for value in transit_prob_given_action_state:
                    if value > 0:
                        v_temp += value*((reward[index])+discount*v_i[index])
                    index += 1
                    
                if v_temp > v_max:
                    v_max = v_temp
                    action_max = action_index
                    
            #update v_previous and v_i
            v_previous[state_index] = v_i[state_index]
            v_i[state_index] = v_max

            value_based_policy.append(action_max)
        
        #Just for printing the value matrix
        # print(np.linalg.norm(v_i - v_previous))
        print(i)
        # draw_square(v_i)
        
        #Add threshold stopping condition
        if np.linalg.norm(v_i - v_previous) <= threshold:
            # draw_square(v_i)
            return v_i,value_based_policy
    return v_i,value_based_policy




if __name__ == "__main__":
    reward = reward_function(8, 10, -10)
    value_function,value_policy = value_function_iteration(0.8,reward,100, 0.01)
    draw_square(value_function)
    draw_square(draw_action(value_policy))