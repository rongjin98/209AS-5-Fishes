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


def value_function_iteration(discount, reward, H, threshold):
    #v_i = np.zeros(len(grid.stateSpace)) #initial value at time-step = 0
    v_previous = np.zeros(len(grid.stateSpace)) #value at time-step = 0
    v_i = reward #value at time-step = 1
    for i in range(H):
        for state_index in range(len(grid.stateSpace)):
            '''
            we need to calculate value function for each state
            '''
            v_temp = []
            for action_index in range(len(grid.actionSpace)):
                '''
                loop through the whole actionspace to find the action to maximize value function
                '''
                transit_prob_given_action_state = grid.transition_probability[action_index][state_index]

                v_s_i = np.sum(np.multiply(transit_prob_given_action_state,reward))
                +np.sum(np.multiply(transit_prob_given_action_state,np.multiply(v_i,discount)))

                v_s_i = round(v_s_i,3)

                v_temp.append(v_s_i)
            #update v_previous and v_i
            v_previous[state_index] = v_i[state_index]
            v_i[state_index] = np.amax(v_temp)
        
        #Just for printing the value matrix
        draw_square(v_i)
        
        #Add threshold stopping condition
        if np.linalg.norm(v_i - v_previous) <= threshold:
            print(np.linalg.norm(v_i - v_previous))
            print(i)
            return v_i
    return v_i

if __name__ == "__main__":
    reward = reward_function(1, 10, -1)
    value_function = value_function_iteration(0.9,reward,100, 5)