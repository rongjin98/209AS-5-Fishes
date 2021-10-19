def draw_square(array,gridSize):
    print("--------------------------")
    for i in range(gridSize):
        sth = []
        for j in range(gridSize):
            sth.append(array[gridSize*i+j])
        print(sth)
    print("--------------------------")
    return None

def make_square(array,gridSize):
    square_matrix = []
    for i in range(gridSize):
        sth = []
        for j in range(gridSize):
            sth.append(array[gridSize*i+j])
        square_matrix.append(sth)
    return square_matrix


def draw_action(array):
    symbol_array = []
    for i in range(len(array)):
        if array[i] == 0:
            symbol_array.append(u'\u2191')
        elif array[i] == 1:
            symbol_array.append(u'\u2193')
        elif array[i] == 2:
            symbol_array.append(u'\u2190')
        elif array[i] == 3:
            symbol_array.append(u'\u2192')
        elif array[i] == 4:
            symbol_array.append(u'\u2298')
    return symbol_array

def get_action_name(index):
        if index == 0: 
            action_name = "Up"
        elif index == 1:
            action_name = "Down"
        elif index == 2:
            action_name = "left"
        elif index == 3:
            action_name = "right"
        elif index == 4:
            action_name = "stay"
        return action_name

def word_descriptions(bool,stateSpace, action_index, new_index):
    if bool == True:
        print("The action picked is: " + get_action_name(action_index)) #TBD
        print("The action is executed correctly")
        print("The new state is now: ", stateSpace[new_index])
    else:
        print("The action picked is: " + get_action_name(action_index)) #TBD
        print("Note: The action is not executed correctly!")
        print("The new state is now: ", stateSpace[new_index])