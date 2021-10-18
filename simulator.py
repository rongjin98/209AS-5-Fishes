import numpy as np
import random
from visualizer import word_descriptions

class Simulator:
    def __init__(self, MDP):

        self.statespace = MDP.stateSpace
        self.actionset = MDP.actionSpace
        self.wind = MDP.wind
        self.observ = MDP.observation
        self.current_state = MDP.current_position
        self.transit_prob = MDP.transition_probability

    #删了，只是让print out的结果看得更清楚
    #ignore this function, need to be deleted 
    


    def update_state(self,action):
        action_index = action
        prob_given_action = self.transit_prob[action_index] #25*25
        for i in range(len(self.statespace)):
            #inital current state should never be in block state in theory
            #initial current state should always within the statespace in theory
            if np.array_equal(self.current_state, self.statespace[i]):
                prob_given_action_state = prob_given_action[i] #1*25
                max_prob = np.amax(prob_given_action_state)
                max_index = np.argmax(prob_given_action_state)
                if random.random() < max_prob:
                    word_descriptions(True,self.statespace,action_index,max_index)
                    return self.statespace[max_index]
                else:
                    successor_when_error = np.delete(np.argwhere(prob_given_action_state > 0),np.argwhere(max_index))
                    roll_a_dice = np.random.choice(successor_when_error) #这个有点hardcode，剩余的各个states的probabili可能是不相同的
                    word_descriptions(False,self.statespace,action_index,roll_a_dice)
                    return self.statespace[roll_a_dice]
        return self.current_state #just in case, if current_state is out of edge
     




