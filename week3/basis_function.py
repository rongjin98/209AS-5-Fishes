import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from gridworld_setup import GridWorld
from double_agents_transition_prob import two_agents_world



class Basis_Function:
    def __init__(self, gridSize, blockSpace, targetSpace, stateSpace, substateSpace): #for two agent gridworld only
        self.gridSize = gridSize
        self.blockSpace = blockSpace
        self.targetSpace = targetSpace
        self.stateSpace = stateSpace
        self.substateSpace = substateSpace
        self.basis_subset = self.createBasisSubset
        self.basis_fullset = self.createBasisFullset
    
    def agent1_x_postion(agent1):
        return agent1[0]
    
    def agent1_y_postion(agent1):
        return agent1[1]
    
    def agent2_x_postion(agent2):
        return agent2[0]
    
    def agent2_y_postion(agent2):
        return agent2[1]

    def agent1_to_nearest_shop(agent1,self):
        dist = 9999
        for shop_ in self.targetSpace:
            dist_temp = np.linalg.norm(agent1-shop_)
            if dist_temp < dist:
                dist = dist_temp
        return dist
    
    def agent2_to_nearest_shop(agent2,self):
        dist = 9999
        for shop_ in self.targetSpace:
            dist_temp = np.linalg.norm(agent2-shop_)
            if dist_temp < dist:
                dist = dist_temp
        return dist
    
    def agent1_to_nearest_block(agent1,self):
        dist = 9999
        for block_ in self.blockSpace:
            dist_temp = np.linalg.norm(agent1 - block_)
            if dist_temp < dist:
                dist = dist_temp
        return dist
    
    def agent2_to_nearest_block(agent2,self):
        dist = 9999
        for block_ in self.blockSpace:
            dist_temp = np.linalg.norm(agent2 - block_)
            if dist_temp < dist:
                dist = dist_temp
        return dist
    
    def agent1_to_agent2(agent1, agent2):
        dist = np.linalg.norm(agent1 - agent2)
        return dist
    
    
    def constant_offset():
        return 1

    @property
    def createBasisSubset(self):
        basis_set = []
        for state_index in self.substateSpace:
            state_ = self.stateSpace[state_index]
            agent1_ = two_agents_world.index_to_position(state_[0],self.gridSize)
            agent2_ = two_agents_world.index_to_position(state_[1],self.gridSize)

            basis_a = Basis_Function.agent1_x_postion(agent1_)
            basis_b = Basis_Function.agent1_y_postion(agent1_)
            basis_c = Basis_Function.agent2_x_postion(agent2_)
            basis_d = Basis_Function.agent2_y_postion(agent2_)

            basis_1 = Basis_Function.agent1_to_nearest_shop(agent1_,self)
            basis_2 = Basis_Function.agent2_to_nearest_shop(agent2_,self)
            basis_3 = Basis_Function.agent1_to_nearest_block(agent1_,self)
            basis_4 = Basis_Function.agent2_to_nearest_block(agent2_,self)
            if basis_3 == 0 or basis_4 == 0:
                #basis_set.append([0, 0, 0, 0, 0, 1])
                basis_set.append([0,0,0,0,0, 0, 0, 0, 0, 1])
            else:
                basis_5 = Basis_Function.agent1_to_agent2(agent1_, agent2_)
                basis_6 = Basis_Function.constant_offset()
                # basis_set.append([basis_1, basis_2, basis_3, basis_4, basis_5, basis_6])
                basis_set.append([basis_a, basis_b, basis_c, basis_d, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6])
            
        return np.array(basis_set)
    
    @property
    def createBasisFullset(self):
        basis_set = []
        for state_ in self.stateSpace:
            agent1_ = two_agents_world.index_to_position(state_[0],self.gridSize)
            agent2_ = two_agents_world.index_to_position(state_[1],self.gridSize)

            basis_a = Basis_Function.agent1_x_postion(agent1_)
            basis_b = Basis_Function.agent1_y_postion(agent1_)
            basis_c = Basis_Function.agent2_x_postion(agent2_)
            basis_d = Basis_Function.agent2_y_postion(agent2_)

            basis_1 = Basis_Function.agent1_to_nearest_shop(agent1_,self)
            basis_2 = Basis_Function.agent2_to_nearest_shop(agent2_,self)
            basis_3 = Basis_Function.agent1_to_nearest_block(agent1_,self)
            basis_4 = Basis_Function.agent2_to_nearest_block(agent2_,self)
            if basis_3 == 0 or basis_4 == 0:
                #basis_set.append([0, 0, 0, 0, 0, 1])
                basis_set.append([0,0,0,0,0, 0, 0, 0, 0, 1])
            else:
                basis_5 = Basis_Function.agent1_to_agent2(agent1_, agent2_)
                basis_6 = Basis_Function.constant_offset()
                #basis_set.append([basis_1, basis_2, basis_3, basis_4, basis_5, basis_6])
                basis_set.append([basis_a, basis_b, basis_c, basis_d, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6])
            
        return np.array(basis_set)


