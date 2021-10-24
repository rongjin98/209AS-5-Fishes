import numpy as np

if __name__ == "__main__":
    state_sapce = np.load('saved_data/stateSpace.npy')
    action_space = np.load('saved_data/actionSpace.npy')
    #[5,6]
    i = 7
    j = 22
    index = i*25+j
    print(action_space[len(action_space)-1])