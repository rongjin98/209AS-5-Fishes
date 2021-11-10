import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from double_agents_function_approximation import non_block_state_index
from gridworld_setup import GridWorld
from double_agents_transition_prob import two_agents_world
from basis_function import Basis_Function
import visualizer


def get_policy_at_state(agent_1_index, agent_2_index, policy_set, double_agents_gridSize):
    double_agents_index = agent_1_index * double_agents_gridSize + agent_2_index
    index_of_actionSpace = policy_set[double_agents_index]
    return index_of_actionSpace


def successor_state_index(agent1_index, agent2_index, action_index, agents_gridSize, gridSize):
    agent1_position = two_agents_world.index_to_position(
        agent1_index, gridSize)
    agent2_position = two_agents_world.index_to_position(
        agent2_index, gridSize)

    action_1 = two_agents_world.index_to_action(action_index[0])
    action_2 = two_agents_world.index_to_action(action_index[1])
    print("The action of agent1 is: ", action_1, ". The action of agent2 is: ", action_2)

    agent1_new = agent1_position + action_1
    agent2_new = agent2_position + action_2

    agent1_index = two_agents_world.position_to_index(agent1_new, gridSize)
    agent2_index = two_agents_world.position_to_index(agent2_new, gridSize)

    index_in_stateSpace = agent1_index*agents_gridSize + agent2_index
    return index_in_stateSpace


def draw_grid_value_at_given_agent1_state(agent1_index, value_set, gridSize, agents_gridSize):
    visualizer.draw_square(
        value_set[agents_gridSize*agent1_index:agents_gridSize*(agent1_index+1)+1], gridSize)
    return


def check_if_in_target(target_set, agent_pos):
    if_in_target = (target_set == agent_pos).all(1).any()
    return if_in_target


if __name__ == "__main__":
    grid = GridWorld([0, 2])
    # initial pos of agent1 and agent2
    two_agents_grid = two_agents_world(grid, [1,0], [0,3]) #change intial state for different simulation results

    gridSize = grid.gridSize
    agents_gridSize = two_agents_grid.agents_gridSize
    target = two_agents_grid.target

    agents_statespace = np.load('saved_data/stateSpace.npy')
    agents_actionspace = np.load('saved_data/actionSpace.npy')
    value_set = np.load('saved_data/value_set.npy')
    policy_set = np.load('saved_data/policy_set.npy')

    agent_1_pos = two_agents_grid.agent_one_position
    agent_2_pos = two_agents_grid.agent_two_position

    print("Intial positions are: ", agent_1_pos, agent_2_pos)

    i = 0
    while check_if_in_target(target, agent_1_pos) == False or check_if_in_target(target, agent_2_pos) == False :
    # i = 0
    # while i < 5:
        agent_1_pos = two_agents_grid.agent_one_position
        agent_2_pos = two_agents_grid.agent_two_position
        agent_1_index = two_agents_world.position_to_index(agent_1_pos, gridSize)
        agent_2_index = two_agents_world.position_to_index(agent_2_pos, gridSize)

        index_of_actionSpace = get_policy_at_state(agent_1_index,agent_2_index,policy_set,agents_gridSize)
        action_index = agents_actionspace[index_of_actionSpace]

        index_of_stateSpace = successor_state_index(agent_1_index, agent_2_index, action_index, agents_gridSize, gridSize)
        new_state = agents_statespace[index_of_stateSpace]

        new_agent_1 = two_agents_world.index_to_position(new_state[0], gridSize)
        new_agent_2 = two_agents_world.index_to_position(new_state[1], gridSize)

        two_agents_grid = two_agents_world(grid, new_agent_1, new_agent_2)
        print("Agent1 is now: ", new_agent_1, ". Agent2 is now: ", new_agent_2)

        i += 1
        if (i > 15):
            break
