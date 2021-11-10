import numpy as np
import math

"""
Right now action is one of the input, 
what if no action is made, but change to the movement of the target
"""

class Bayes:
    def __init__(self, MDP, initial_bel, action): 
        """
        1. Initial_bel contains all the time history
           ---Initial_bel should be equivalent to Bel_t_+
           ---Note Bel_0_+ = [0 0 0 ... 1 .. 0 0 0], since no action has been taken yet
           ---Bel_t+1_+ will be used as the initial_bel for next time step/before next action taken
        2. The current actual position specified by MDP is used as the GroundTruthPosition after the taken action
           ---GroundTruthPosition is only used for observation calculation
        3. The *action* here is the very latest action taken
        4. Transition Probability can be accessed by using the non-zero entries of the Initial_bel and the action
        
        This structure helps to integrate with the simulator
        """
       
        #Therefore, Prior Belief should tell what will happen when take that action
        #Posterior Belief should tell what will happen when take an obs after that action
        self.init_system(MDP)

        self.action = action
        self.bel_0 = initial_bel #Bayes need to keep a track on time history
        self.belief_prior = self.get_belief_prior             # pr(s|*)
        self.belief_posterior = self.get_belief_posterior     # pr(s|o, *)
    
    def init_system(self, MDP):
        self.ground_truth_position = MDP.current_position #get the actual position
        self.target = MDP.target                          #initialize target

        # initialize system
        self.stateSpace = MDP.stateSpace
        self.blockSpace = MDP.blockSpace
        self.transition_pr = MDP.transition_probability
        self.gridSize = MDP.gridSize
        self.obs = self.getObservation                    #rounded observation at groundtruthstate

        self.actualObs = self.getActualObs                # A 25*1 observation map without rounding
    
    @property
    def getActualObs(self):
        actual_obs = []
        for state_ in self.stateSpace:
            obs_ = Bayes.calObs(self.target, state_)
            actual_obs.append(obs_) 
        return np.array(actual_obs)

    @property
    def getObservation(self):
        """
        This function should update current observation

        return observation
        """
        h = Bayes.calObs(self.target, self.ground_truth_position)
        h = Bayes.ceiling_or_flooring(h)
        return h
    
    
    @property
    def get_belief_prior(self):
        """
        Get pr(s|*)
        """
        possible_position_index = []
        for i in range(len(self.bel_0)):
            if self.bel_0[i] != 0:
                possible_position_index.append(i)

        belief_prior= np.zeros(len(self.bel_0))
        for index_ in possible_position_index:
            tran_pr = self.transition_pr[self.action][index_] 
            belief_prior += tran_pr * self.bel_0[index_]
        return belief_prior
    
    @property
    def get_belief_posterior(self):
        """
        Get the p(o|s), due to the floor and ceil rounding, we can guarante that the floored observation map and the 
        ceiled observation map will not have any overlapped grid for a specific observation
        """
        lower_bound_obs_map = Bayes.FloorObs(self.stateSpace,self.blockSpace,self.target)
        upper_bound_obs_map = Bayes.CeilObs(self.stateSpace,self.blockSpace,self.target)
        pr_lower_bound, pr_upper_bound = Bayes.floor_ceil_possibility(self.actualObs)

        pr_obs_state = np.zeros(len(self.stateSpace)) #25*1
        for i in range(len(lower_bound_obs_map)):
            if self.obs == lower_bound_obs_map[i]:
                pr_obs_state[i] = pr_lower_bound[i]

        for j in range(len(upper_bound_obs_map)):
            if self.obs == upper_bound_obs_map[j]:
                pr_obs_state[j] = pr_upper_bound[j]
        
        """
        Get belief_posterior
        """
        temp_ = np.multiply(pr_obs_state, self.belief_prior)
        belief_posterior = temp_/np.sum(temp_)
        return belief_posterior
        

    
    def calObs(target, position):
        """
        Calculate observation

        @param position - an array
        return np.float
        """
        RS, RD = target
        distance2RS = np.linalg.norm(position - RS)
        distance2RD = np.linalg.norm(position - RD)
        if distance2RS == 0 or distance2RD == 0:
            h = 0
        else:
            h = 2 / (1 / distance2RS + 1 / distance2RD)
        return h
    
    def ceiling_or_flooring(h):
        if np.random.rand() < np.ceil(h) - h:
            return np.floor(h)
        else:
            return np.ceil(h)
    
    def floor_ceil_possibility(actual_obs):
        """
        Generate two matrices:
        1. Probability of flooring 25*1
        2. Probability of ceiling  25*1

        Cooperate with actualObs, FloorObs and CeilObs for calculating P(o|s)
        """
        floor = []
        ceil = []
        for obs_ in actual_obs:
            pr_floor = np.ceil(obs_)-obs_
            pr_ceil = 1 - pr_floor
            pr_floor = np.round(pr_floor,3)
            pr_ceil = np.round(pr_ceil,3)
            floor.append(pr_floor)
            ceil.append(pr_ceil)
        return floor, ceil
    
    def FloorObs(stateSpace, blockSpace, target):
        """
        The observation map when always flooring
        """
        obs_1 = []
        for state_ in stateSpace:
            if_block = (blockSpace == state_).all(1).any()
            if if_block == True:
                obs_ = 9.9 #Trivial Value, blockSpace will never be included
            else:
                obs_ = Bayes.calObs(target,state_)
                obs_ = np.floor(obs_)
            obs_1.append(obs_)
        return np.array(obs_1)
    
    def CeilObs(stateSpace, blockSpace, target):
        """
        The observation map when always ceiling
        """
        obs_1 = []
        for state_ in stateSpace:
            if_block = (blockSpace == state_).all(1).any()
            if if_block == True:
                obs_ = 9.9 #Trivial Value, blockSpace will never be included
            else:
                obs_ = Bayes.calObs(target,state_)
                obs_ = np.ceil(obs_)
            obs_1.append(obs_)
        return np.array(obs_1)
    
    
    def index_to_position(index,gridSize): #helper function
        first_pos = math.floor(index/gridSize)
        second_pos = index % gridSize
        pos = [first_pos, second_pos]
        return np.array(pos)
    
    def position_to_index(position,gridSize): #helper function
        index = position[0]*gridSize + position[1]
        return index
    

    
    
    
    

        

