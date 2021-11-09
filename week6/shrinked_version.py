import numpy as np

class three_grid:
    def __init__(self):
        self.stateSpace = np.array([[0,0],[0,1],[0,2]])
        self.action = np.array([[0,-1],[0,1],[0,0]]) #left,right,stay
        self.wind = 0.25
        self.observation = self.ge

