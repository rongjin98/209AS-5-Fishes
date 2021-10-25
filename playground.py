import numpy as np
from double_agents_function_approximation import non_block_state_index, successor_state_index
from gridworld_setup import GridWorld
from double_agents_transition_prob import two_agents_world
from basis_function import Basis_Function
import visualizer
import double_agents_function_approximation
import time

'''
Just for debugging purpose
'''
grid = GridWorld([0,2])
two_agent_grid = two_agents_world(grid,[0,2],[4,4])
reward_map = two_agent_grid.reward_map
count = 0
index_count = 0

non_block = non_block_state_index(reward_map)
print(non_block.shape)
basis_function = Basis_Function(grid.gridSize,grid.blockSpace, grid.target, two_agent_grid.two_stateSpace, non_block)
print(basis_function.basis_set.shape)




# for j in set:
#     state_ = two_agent_grid.two_stateSpace[j] #231
#     agent1 = two_agents_world.index_to_position(state_[0],5)
#     agent2 = two_agents_world.index_to_position(state_[1],5)
#     print(agent1)
#     print(agent2)
#     print("========================")




################Successor_state_index & available_action_set Test################
# state_ = two_agent_grid.two_stateSpace[523] #231
# agent1 = two_agents_world.index_to_position(state_[0],5)
# agent2 = two_agents_world.index_to_position(state_[1],5)
# print(agent1)
# print(agent2)

# action_set = np.load('saved_data/Available_action_set.npy', allow_pickle = True)
# action_space = two_agent_grid.two_actionSpace
# action_ = action_set[523]

# for index in action_:
#     print(action_space[index])
#     action_1 = two_agents_world.index_to_action((action_space[index])[0])
#     action_2 = two_agents_world.index_to_action((action_space[index])[1])
#     successor_state_index = double_agents_function_approximation.successor_state_index(state_[0], state_[1], action_space[index], 25, 5)
#     ss_ = two_agent_grid.two_stateSpace[successor_state_index]
#     ss_1 = two_agents_world.index_to_position(ss_[0],5)
#     ss_2 = two_agents_world.index_to_position(ss_[1],5)
#     print("The first action is: ", action_1, " The resulting state1 is: ", ss_1)
#     print("The second action is: ", action_2, " The resulting state2 is: ", ss_2)
