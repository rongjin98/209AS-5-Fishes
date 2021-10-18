import numpy as np
import random

'''
pe, Rd, Rs, Rw, discount, Time Horizon
'''

def value_function_iteration(discount, reward, H, threshold, MDP):
    v_previous = np.random.rand(len(MDP.stateSpace)) #initial value at time-step = 0
    v_i = np.zeros(len(MDP.stateSpace)) #value at time-step = 0x
    available_action_set = MDP.get_available_actions()
    for i in range(H):
        value_based_policy = []
        for state_index in range(len(MDP.stateSpace)):
            '''
            we need to calculate value function for each state
            '''
            v_max = -999
            action_max = -999
            for action_index in available_action_set[state_index]:
                '''
                loop through the whole actionspace to find the action to maximize value function
                '''
                transit_prob_given_action_state = MDP.transition_probability[action_index][state_index]
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
        # draw_square(v_i)
        
        #Add threshold stopping condition
        if np.linalg.norm(v_i - v_previous) <= threshold:
            # draw_square(v_i)
            return v_i,value_based_policy
    return v_i,value_based_policy
