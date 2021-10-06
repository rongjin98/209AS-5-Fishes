class GridWorld:
    def __init__(self):
        self.state = self.createState
        self.action = self.createAction
        self.probability = self.createProbability
        self.observation = self.createObservation

    @property
    def createState(self):
        NotImplemented
    
    @property
    def createProbability(self):
        NotImplemented
    
    @property
    def createAction(self):
        NotImplemented
        
    @property
    def createObservation(self):
        NotImplemented
        