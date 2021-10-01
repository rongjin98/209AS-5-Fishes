# This is the class of Girdworld
# It should define the Action, State, Transition probability, and the Observation.
# More details about the formulas of above concept will show in there functions

import random


class Gridworld:
    #TODO
    # need to add method to set h and O

    def __init__(self, c_num:int, r_num:int, obstacles_list:list, IC_list:list, robot_pos:tuple, pe:float) -> None:
        # c_num is the colum number of the map 
        # r_num is the row number of the map
        # In this map, (x,y) X axis is horizontal right direction, and Y axis is vertical up direction
        # (0,0) is the left bottom corner
        # obstacles_list and IC_list are list of postion as (x,y) to set the obstacles and ice cream on the map
        # robot_pos is the postion of the initial postion where the robot locate at the map. Ex: (0,0)
        # pe is the error probability when a action can goes wrong.

        # the map will use char type to store information. 
        # '1' means a accessibly grid
        # '0' means a obstacle grid
        # 'i' means a ice cream
        # 'r' means robot itself
        print(pe)

        self.pe = pe
        self.robotPos = robot_pos
        self.obstacles = obstacles_list
        self.IceCreams = IC_list
        self.cNum = c_num
        self.rNum = r_num

        self.preRobotPos = robot_pos                # this is used to make robot postion update easier and faster
        
        # initial the default map
        # TODO THERE IS A ISSUES ABOUT (X,Y) COORDINATE
        self.map = [['1' for i in range(r_num)] for j in range(c_num)]

        # set all obstacles
        for e in obstacles_list:
            self.map[e[0]][e[1]] = '0'
        
        # set all ice cream
        for e in IC_list:
            self.map[e[0]][e[1]] = 'i'
        
        # set the robot postion
        self.map[robot_pos[0]][robot_pos[1]] = 'r'

        # set the action space
        self.actionSpace = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
        # there are 5 possible actions
        # in order they are NoMove, Up, Down, Right, Left
        #                     0     1     2     3     4


    def getMap(self) -> list:
        # this will return the current map which is the state
        # it should be a list of list which is a 2-D array
        return self.map

    def updateMap(self, opts:int) -> list:
        # this used to update map to the newest map when some element in the map changed
        # opts stands for what element changed
        # 0 stands for robot postion changed

        #TODO
        # add more update reason in feature if necessary

        # if the update reason is robot postion changed
        if 0 == opts:
            # remove the previous robot postion from 'r' to '1'
            self.map[self.preRobotPos[0]][self.preRobotPos[1]] = '1'

            # if previous postion is on ice cream
            # put ice scream tempearely
            # TODO this step need more information 
            for e in self.IceCreams:
                if e[0] == self.preRobotPos[0] and e[1] == self.preRobotPos[1]:
                    self.map[self.preRobotPos[0]][self.preRobotPos[1]] = 'i'

            # set the new robot postion
            self.map[self.robotPos[0]][self.robotPos[1]] = 'r'
            # set all ice cream


    def takeAction(self, action:int, pe:float) -> tuple:
        # This function is to let robot make a movement with (1-pe) probability success
        # And when action failed, it will get a randomly movement from 4 alternative direction
        # the return value is the new postion after robot took a action
        randomVal = random.random()
        movement = (0,0)
        
        print(action)
        
        # since randomVal in range [0,1), thus when it less than pe, we treat it as success
        if randomVal < (1-pe):
            # if success, movement is the  target direction
            movement =  self.actionSpace[action]
        else:
            # if failed, movement pick a direction from 4 alternative directions
            randomNum = random.randint(1,4)
            action = (action+randomNum)%5
            movement = self.actionSpace[action]
        print(action)
        print("==============")
    
        # then check after movement, is the new postion valid
        # if valid, move the robot to the new postion, else let the robot NoMove
        newPos = (self.robotPos[0] + movement[0], self.robotPos[1] + movement[1])

        # first check is it out of the boundary
        if  newPos[0] < 0   or  newPos[0] >= self.cNum  or \
            newPos[1] < 0   or  newPos[1] >= self.rNum:
            # if out of the boundary, NoMove return the previous postion
            return self.robotPos, 0

        # then check is it move into an obstacle
        for e in self.obstacles:
            if e[0] == newPos[0] and e[1] == newPos[1]:
                # if moved into a obstacle, NoMove return the previous postion
                return self.robotPos, 0
        
        # Reach here means, the new postion is valid.
        # Then update the new postion and return
        self.preRobotPos = self.robotPos
        self.robotPos = newPos
        return self.robotPos, action