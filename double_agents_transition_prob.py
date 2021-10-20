#2-agents transition probability
import numpy as np
from numpy.core.records import array
from gridworld_setup import GridWorld
from numpy.lib.function_base import append
from visualizer import draw_square, make_square
import math
import time

'''
Action = [up down left right stay]
'''

class two_agents_world:
    def __init__(self,MDP,pos1,pos2):
        self.gridSize = MDP.gridSize
        self.agents_gridSize = self.gridSize * self.gridSize
        self.blockSpace = MDP.blockSpace
        self.actionSet = MDP.actionSpace
        self.stateSet = MDP.stateSpace
        self.agent_one_position = np.array(pos1)
        self.agent_two_position = np.array(pos2)
        self.wind = MDP.wind

        self.two_actionSpace = self.createAction #25
        self.two_stateSpace = self.createSpace #625
        #self.successor_state_given_state = self.createSS #625*625
        self.transition_probability = self.createTransition #25*625*625

    @property
    def createAction(self):
        actionSpace = []
        for i in range(len(self.actionSet)):
            for j in range(len(self.actionSet)):
                temp = [i, j]
                actionSpace.append(temp)
        return actionSpace
    
    @property
    def createSpace(self):
        stateSpace = []
        for i in range(len(self.stateSet)):
            for j in range(len(self.stateSet)):
                temp = [i,j]
                stateSpace.append(temp)
        return np.array(stateSpace)
    
    #This property can be ignored, createSS returns a 625*625/(25*25*25*25) matrix, represents the available successor states for each current state 
    # it is kept here for the structural reference of a transition probability matrix
    @property
    def createSS(self):
        ss_given_state =[]
        for i in range(self.agents_gridSize):
            temp = []
            for j in range(self.agents_gridSize):
                temp2 = np.zeros([self.agents_gridSize, self.agents_gridSize])
                position_state = two_agents_world.index_matrix_to_position_matrix(self.two_stateSpace,self.gridSize) #change the expression of statespace
                position_state = np.array(make_square(position_state,self.agents_gridSize)) #reshape the statespace from 625 to 25*25
                state_agent_1 = position_state[i][j][0]
                state_agent_2 = position_state[i][j][1]
                ss_1,ss_2 = two_agents_world.get_ss_each_agent(state_agent_1,state_agent_2,self.stateSet,self.blockSpace)
                index_ss1 = two_agents_world.get_non_zero_index(ss_1) #get non-zero entry inside the successor state array
                index_ss2 = two_agents_world.get_non_zero_index(ss_2) 
                '''
                Map the two 25*1 array to an 25*25 array, row is for agent1, column is for agent 2

                Thus index would be:
                  1. Agent1 -- temp[i][index_1]
                  2. Agent2 -- temp[index2][i]
                Mark the posiitons above as 1
                
                Note: There exists the condition which two agents can reach the same position. We need to mark such postion as 2,
                for transition probability calculation
                '''
                for index_1 in index_ss1:
                    temp2[i][index_1] += 1
                
                for index_2 in index_ss2:
                    temp2[index_2][j] += 1
                temp.append(temp2)
            ss_given_state.append(temp)
        return np.array(ss_given_state)
    

    #The Probability Transition Matrix has the size of 25*625*625 (25*25^2*25^2)
    @property
    def createTransition(self):
        #action_ in two_actionSpace is an array with size of 2, 
        #eg action_ = [0,0], which means both agents take "forward" action
        transit_prob = [] #25*625*625
        for action_ in self.two_actionSpace: 
            temp_layer_1 = []  #625*625
            action_agent1 = self.actionSet[action_[0]]
            action_agent2 = self.actionSet[action_[1]]
            for i in range(self.agents_gridSize):
                temp_layer_2 = [] # 25* 625
                for j in range(self.agents_gridSize):
                    temp_layer_3 = np.zeros([self.agents_gridSize,self.agents_gridSize])
                    position_state = two_agents_world.index_matrix_to_position_matrix(self.two_stateSpace,self.gridSize)
                    position_state = np.array(make_square(position_state,self.agents_gridSize))
                    current_state_agent_1 = position_state[i][j][0]
                    current_state_agent_2 = position_state[i][j][1]

                    transition_at_agent1, transition_at_agent2 = two_agents_world.get_transition_each_agent_state_action(current_state_agent_1,
                    current_state_agent_2, action_agent1, action_agent2, self.stateSet, self.actionSet, self.blockSpace, self.wind, self.gridSize)

                    index_trsit_1 = two_agents_world.get_non_zero_index(transition_at_agent1)
                    index_trsit_2 = two_agents_world.get_non_zero_index(transition_at_agent2)

                    '''
                    Refer to createSS() for structural interpretation
                    '''
                    for index_1 in index_trsit_1:
                        for index_2 in index_trsit_2:
                            temp_layer_3[index_1][index_2] = transition_at_agent1[index_1]*transition_at_agent2[index_2]
                    
                    temp_layer_2.append(temp_layer_3)
                temp_layer_1.append(temp_layer_2)
            transit_prob.append(temp_layer_1)
        return np.array(transit_prob)
    
    #create two transition probabilities of 25*1 arrays for agent1 and agent2 
    def get_transition_each_agent_state_action(current_state_1, current_state_2, action_agent1, action_agent2, stateSet, actionSet, blockSet, wind, gridSize): #helper function1 for createTransition()
        ss_mark_1, ss_mark_2 = two_agents_world.get_ss_each_agent(current_state_1,current_state_2,stateSet,blockSet) # 2 25*1 array

        available_action_agent_1 = two_agents_world.get_available_action(action_agent1, current_state_1,stateSet,actionSet,blockSet)
        available_action_agent_2 = two_agents_world.get_available_action(action_agent2,current_state_2,stateSet,actionSet,blockSet)

        non_zero_index_1 = two_agents_world.get_non_zero_index(ss_mark_1)
        non_zero_index_2 = two_agents_world.get_non_zero_index(ss_mark_2)

        current_state_1_index = two_agents_world.position_to_index(current_state_1,gridSize)
        current_state_2_index = two_agents_world.position_to_index(current_state_2, gridSize)

        transit_p1 = np.zeros(len(ss_mark_1))
        transit_p2 = np.zeros(len(ss_mark_2))


        # print("Avail_1 : ", available_action_agent_1)

        # print("Avail_2 : ", available_action_agent_2)
        # print("Agent2: ", action_agent2)
        if_action_available_1 = (available_action_agent_1 == action_agent1).all(1).any(0)
        if_action_available_2 = (available_action_agent_2 == action_agent2).all(1).any(0)


        if if_action_available_1 == True:
            new_state_1 = current_state_1 + action_agent1
            new_state_index_1 = two_agents_world.position_to_index(new_state_1,gridSize)
            transit_p1[new_state_index_1] = 1 - wind #make sure wind is not equal to 0
            for index_1 in non_zero_index_1:
                if ss_mark_1[index_1] == 1 and transit_p1[index_1] == 0:
                    transit_p1[index_1] = wind/(len(available_action_agent_1)-1) #refer to gridworld_setup.py -- create_transition_probability -- eg.3
        else:
            transit_p1[current_state_1_index] = (1 - wind) + wind/len(available_action_agent_1)
            for index_11 in non_zero_index_1:
                if ss_mark_1[index_11] == 1 and transit_p1[index_11] == 0: #Avoid overwrite
                    transit_p1[index_11] = wind/len(available_action_agent_1)

        if if_action_available_2 == True:
            new_state_2 = current_state_2 + action_agent2
            new_state_index_2 = two_agents_world.position_to_index(new_state_2,gridSize)
            transit_p2[new_state_index_2] = 1 - wind
            for index_2 in non_zero_index_2:
                if ss_mark_2[index_2] == 1 and transit_p2[index_2] == 0:
                    transit_p2[index_2] = wind/(len(available_action_agent_2)-1)
        else:
            transit_p2[current_state_2_index] = (1-wind) + wind/len(available_action_agent_2)
            for index_22 in non_zero_index_2:
                if ss_mark_2[index_22] == 1 and transit_p2[index_22] == 0:
                    transit_p2[index_22] = wind/len(available_action_agent_2)


        return transit_p1, transit_p2

    def get_available_action(action, current_state, stateSet, actionSet, blockSet): #helper function2 for createTransition()
        #get the available action from the given actionset at particular current state
        available_action_at_state = []
        if_block = (blockSet == current_state).all(1).any(0)
        if if_block == True:
            available_action_at_state.append([0,0]) #only stay action when at blockstate
        else:
            for action_ in actionSet:
                next_state = current_state + action_
                if_in_block = (blockSet == next_state).all(1).any(0)
                if_in_bound = (stateSet == next_state).all(1).any(0)
                if if_in_block == False and if_in_bound == True:
                    available_action_at_state.append(action_)
        return available_action_at_state
    

    def get_ss_each_agent(current_state_1, current_state_2, stateSet, blockSet): #helper function for createSS()
        current_state_1 = np.array(current_state_1)
        current_state_2 = np.array(current_state_2)

        if_block1 = (blockSet == current_state_1).all(1).any(0)
        if_block2 = (blockSet == current_state_2).all(1).any(0)

        ss_1 = []
        ss_2 = []
        for state_ in stateSet:
            if if_block1 == True:
                ss_1.append(0)
            else:
                if_block3 = (blockSet == state_).all(1).any(0)
                dist1 = np.linalg.norm(current_state_1 - state_)
                if if_block3 == False and dist1 <= 1:
                    ss_1.append(1)
                else:
                    ss_1.append(0)

            if if_block2 == True:
                ss_2.append(0)
            else:
                if_block4 = (blockSet == state_).all(1).any(0)
                dist2 = np.linalg.norm(current_state_2 -  state_)
                if if_block4 == False and dist2 <= 1:
                    ss_2.append(1)
                else:
                    ss_2.append(0)
        
        return np.array(ss_1),np.array(ss_2)


    def get_non_zero_index(input_array): #helper function for createSS()
        index_array = []
        for i in range(len(input_array)):
            if input_array[i] != 0:
                index_array.append(i)

        return np.array(index_array)

    
    def index_to_position(index,gridSize):
        first_pos = math.floor(index/gridSize)
        second_pos = index % gridSize
        pos = [first_pos, second_pos]
        return np.array(pos)
    
    def position_to_index(position,gridSize):
        index = position[0]*gridSize + position[1]
        return index
    
    def index_matrix_to_position_matrix(index_matrix,gridSize):
        position_matrix = []
        for index_array in index_matrix:
            temp = []
            #print(index_array)
            for index in index_array:
                pos1 = two_agents_world.index_to_position(index,gridSize)
                temp.append(pos1)
            position_matrix.append(temp)
        return np.array(position_matrix)

         



if __name__ == "__main__":
    grid = GridWorld([0,2])

    start = time.time()
    two_agent_grid = two_agents_world(grid,[0,2],[0,3])
    end = time.time()

    print(two_agent_grid.transition_probability[0][0][0])

#     print("It took ", end - start, " seconds to complete the initialization")
    # ss_1,ss_2 = two_agents_world.get_ss_each_agent([4,4],[0,0],grid.stateSpace,grid.blockSpace)
    # draw_square(ss_1,5)
    # draw_square(ss_2,5)

    # index1 = two_agents_world.get_non_zero_index(ss_1)
    # index2 = two_agents_world.get_non_zero_index(ss_2)
    #draw_square(two_agent_grid.two_actionSpace,two_agent_grid.gridSize)
    
    # transit_p1,transit_p2 = two_agents_world.get_transition_each_agent_state_action([0,0],[4,1],np.array([0,1]),np.array([0,1]),grid.stateSpace, grid.actionSpace, grid.blockSpace, grid.wind, grid.gridSize)
    # print(two_agents_world.get_non_zero_index(transit_p1))
    # print(two_agents_world.get_non_zero_index(transit_p2))
    # print(np.sum(transit_p1)+np.sum(transit_p2))


    # transit_p2 = transit_p2.reshape((25,1))
    # transit_p1 = transit_p1.reshape((25,1))
    # transit_p2 = np.transpose(transit_p2)

    # transit_total = transit_p1 @ transit_p2
    # print(np.sum(transit_total))

    # print(transit_total.shape)

    # reward_map1 = grid.reward_function()
    # reward_map1 = reward_map1.reshape((25,1))

    # reward_map2 = grid.reward_function()
    # reward_map2 = reward_map2.reshape((25,1))
    # reward_map2 = np.transpose(reward_map2)
    # print(reward_map2.shape)

    # reward_total = reward_map1 @ reward_map2

    # transit_total = transit_total.flatten()
    # reward_total = reward_total.flatten()
    # transit_total = transit_total.reshape((625,1))
    # reward_total = reward_total.reshape((625,1))
    # transit_total = np.transpose(transit_total)

    # value = transit_total @ reward_total
    # value = value.flatten()
    # print(value)


    # draw_square(transit_p1,5)
    # draw_square(transit_p2,5)
    # draw_square(two_agent_grid.two_stateSpace,two_agent_grid.agents_gridSize)
    # pos_state = two_agents_world.index_matrix_to_position_matrix(two_agent_grid.two_stateSpace,two_agent_grid.gridSize)
    # print(two_agent_grid.two_stateSpace.shape)
    # pos_state = np.array(make_square(pos_state,two_agent_grid.agents_gridSize))
    # print(pos_state[24][0][1])

    # print(two_agent_grid.successor_state_given_state.shape)
    # print(two_agent_grid.successor_state_given_state[0][0])
    # print(two_agent_grid.transition_probability.shape)
    # print(two_agent_grid.transition_probability[1][0][0]) #transition probability given "foward" action, given agent1 and agent2 both at [0 0]
    
    
    '''
    pos_state --- pos_state[0] --> fix agent1
              --- pos_state[x][0] --> fix agent2
              --- pos_state[x][y][0] --> agent1
              --- pos_state[x][y][1] --> agent2
    '''