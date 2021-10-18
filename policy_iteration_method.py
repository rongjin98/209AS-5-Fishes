import numpy as np
import random

def setup_random_policy(availble_actions,MDP):
    policy_map =[]
    for i in range(len(MDP.stateSpace)):
        viable_action_set = availble_actions[i]
        pick_random_action = np.random.choice(viable_action_set)
        policy_map.append(pick_random_action)
    return np.array(policy_map)




def policy_function_iteration(discount, reward, H, MDP):
    available_action_set = MDP.get_available_actions()
    previous_policy = setup_random_policy(available_action_set,MDP)
    initial_policy = setup_random_policy(available_action_set,MDP)

    v_evaluate  = np.zeros(len(MDP.stateSpace))
    epoch = 0

    while np.array_equal(previous_policy,initial_policy) == False and epoch < H:
        new_policy = []
        #policy evaluation:
        for state_index in range(len(MDP.stateSpace)):
            transit_prob_given_action_state = MDP.transition_probability[initial_policy[state_index]][state_index]
            policy_evaluation = 0
            index1 = 0
            for value in transit_prob_given_action_state:
                if value > 0:
                    policy_evaluation += value*((reward[index1])+discount*v_evaluate[index1])
                index1 += 1
            v_evaluate[state_index] = policy_evaluation
        
        #policy refinement:
        for state_index2 in range(len(MDP.stateSpace)):
            q_max = -999
            action_max = -999
            for action_index in available_action_set[state_index2]:
                transit_prob_given_action_state2 = MDP.transition_probability[action_index][state_index2]

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
    return initial_policy
