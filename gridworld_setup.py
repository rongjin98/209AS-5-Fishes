import numpy as np


class GridWorld:
    def __init__(self):
        self.pe = .25
        self.numActions = 5
        self.gridSize = 5
        self.block = np.array([(1,1), (1,2), (3,1), (3,2)])
        self.target = np.array([(2,0), (2,2)])
        self.position = np.array([2,4])
        
        self.state = self.createState
        self.action = self.createAction
        self.probability = self.createProbability
        self.observation = self.createObservation

    @property
    def createState(self):
        state_list = []
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                temp_state = [i,j]
                state_list.append(temp_state)
        return np.array(state_list)
    
    @property
    def createProbability(self):
        '''
        direction:
        1 2 3 4 5
        left right up down stay
        '''
        actionSpace = self.action
        blockSpace = self.block
        stateSpace = self.state
        
        
        prob = []
        #the probability space should be in size of Ns^2*Na(25^2*5), yet for this particular gridworld
        #since the robot is only allowed to move for 1 step each time, Ns*Na is enough
        #It can be extended to (25^2*5) if necessary in the simulator
        for states in stateSpace: 
            transition_pr = []
            if_block = (blockSpace == states).all(1).any()
            #assume equally possibilities of selecting an action within available action set
            #count for how many available actions at one specific state
            for actions in actionSpace: #action [up,down,left,right,stay]
                nextState = np.array(states+actions)
                if if_block == True:
                    transition_pr.append(0)
                    #if the robot is in block state which is impossible, 
                    #all of its transition possibilities are zero
                elif if_block == False:
                    if_in_boundary = (stateSpace == nextState).all(1).any()
                    if_in_block = (blockSpace == nextState).all(1).any()
                    if if_in_boundary == True and if_in_block == False:
                        transition_pr.append(1)
                    else:
                        transition_pr.append(0)
            if(if_block == True):
                prob.append(transition_pr)
            else:
                pr = 1.0/sum(transition_pr)#equal possibilities for every available actions
                for i in range(len(transition_pr)):
                    if(transition_pr[i] == 1):
                        transition_pr[i] = pr
                prob.append(transition_pr)
        return np.array(prob)
    
    
    @property
    def createAction(self):
        forward = [-1, 0]
        backward = [1, 0]
        left = [0 ,-1]
        right = [0 ,1]
        stay= [0,0]

        return np.array([forward, backward, left, right, stay])
        
    @property
    def createObservation(self):
        position = self.position
        RS, RD = self.target
        distance2RS = np.linalg.norm(position - RS)
        distance2RD = np.linalg.norm(position - RD)
        h = 2 / (1 / distance2RS + 1 / distance2RD)
        
        if np.random.rand() < np.ceil(h) - h:
            return np.floor(h)
        else:
            return np.ceil(h)
    
    def todo(self, *args, **kwargs):
        print(probability)
        print("NotImplemented")
    
    
"""
testing GridWorld
"""   
if __name__ == "__main__":
    grid = GridWorld()
   #print(grid.state)
    #for i in range(len(grid.probability)):
        #print(grid.probability[i], grid.state[i])
    print(grid.probability)
    #print(grid.observation)