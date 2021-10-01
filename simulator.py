#!/user/bin/env python
# -*- coding:utf-8 -*-

import re
import sys, time, os, pygame
from pygame.display import iconify
from pygame.locals import *
from gridworld import Gridworld
from robot import Robot

GRID_SIZE = 50
LINE_SIZE = 3

# this dict is used to map the arrow input to the robot move direction
        # NoMove, Up, Down, Right, Left
        #   0     1     2     3     4
ROTATIONS = {pygame.K_RETURN: 0, pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_RIGHT: 3, pygame.K_LEFT: 4}
NUM2DIRECTION = {0:"NoMove", 1:"Up", 2:"Down", 3:"Right", 4:"Left"}

# this simulator will let the robot running on the gridworld
class Simulator:
    def __init__(self, gridWorld:Gridworld, width, height) -> None:
        # construction function need one Gridworld object to simulate
        # width and height is the width and height of the simulator window

        self.width = width
        self.height = height

        # initial the simulator window configs
        pygame.init()
        pygame.display.set_caption("Simulator")
        self.screen = pygame.display.set_mode((width+200, height))

        # set the simulator objects, include gridworld(Environment) and robot
        self.gridWorld = gridWorld
        self.robot = Robot()
        self.isAuto = False             # this variable used as a flag to set is the robot take action automatic or manually
                                        # by default, it's movement is be manually controlled
        
        # set the font
        self.myFont = pygame.font.SysFont("arial", 20)

    def loop(self) -> None:
        actionText = "Original Action: None"
        actualActionText = "Actually Action: None"

        # this is the main loop that in a while loop to keep the simulator running
        while True:
            action = None                   # default direction is NoMove
            actualAction = None             # to store the actual action after took an action

            # each loop, the simulator redraw the whole screen to update the changes
            self.drawScreen()

            # then get event and process it
            for event in pygame.event.get():
                # got an quit event, then close the simulator and exit the system
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                #TODO
                # Add more hot key for this simulator

                # then check is there any "KEYDOWN" events
                if event.type == KEYDOWN:
                    # "a" key is used to switch between auto/manual
                    if event.key == K_a:
                        self.isAuto = not self.isAuto

                    # to check if is movement keys
                    if self.isMovementKeys(event.key):
                        action = ROTATIONS[event.key]

            if self.isAuto:
                action = self.robot.makeDecision()
                _, actualAction = self.gridWorld.takeAction(action, self.gridWorld.pe)
                actionText = "Original Action: " + NUM2DIRECTION[action]
                actualActionText = "Actually Action: " + NUM2DIRECTION[actualAction]
            else:
                if action != None:
                    _, actualAction = self.gridWorld.takeAction(action, 0)
                    actionText = "Original Action: " + NUM2DIRECTION[action]
                    actualActionText = "Actually Action: " + NUM2DIRECTION[actualAction]

            self.screen.fill(pygame.Color("black"), (self.width+5,0, self.width+200, self.height))
            self.screen.blit(self.myFont.render(actionText, True, pygame.Color("white"), pygame.Color("black")), (self.width+10, 5))
            self.screen.blit(self.myFont.render(actualActionText, True, pygame.Color("white"),pygame.Color("black")), (self.width+10, 55))

            self.gridWorld.updateMap(0)

            # this is used to control the simulate speed
            pygame.time.wait(10)
            pygame.display.update()

    def drawScreen(self) -> None:
        map = self.gridWorld.getMap()
        cNum = self.gridWorld.cNum
        rNum = self.gridWorld.rNum
        x = y = 0                       # this used to store really x, y in the screen
        px = py = 0                     # this used to store x, y pixel in the screen
        color = pygame.Color("black")

        for i in range(cNum):
            for j in range(rNum):
                
                
                # calculate the rectangle
                # and do an matrix transform
                x = i
                px = x*GRID_SIZE
                y = rNum - j - 1
                py = y*GRID_SIZE

                if '1' == map[i][j]:
                    color = pygame.Color("white")
                elif '0' == map[i][j]:
                    color = pygame.Color("darkgrey")
                elif 'i' == map[i][j]:
                    color = pygame.Color("green3")
                else:
                    color = pygame.Color("darkorchid1")


                pygame.draw.rect(self.screen, color, ((px,py),(GRID_SIZE,GRID_SIZE)), width = 0)
        
        for i in range(cNum):
            pygame.draw.line(self.screen, pygame.Color("black"), (i*GRID_SIZE,0), (i*GRID_SIZE, rNum*GRID_SIZE), 3)
        for j in range(rNum):
            pygame.draw.line(self.screen, pygame.Color("black"), (0, j*GRID_SIZE), (cNum*GRID_SIZE, j*GRID_SIZE), 3)
        pygame.display.update()

    def isMovementKeys(self, k) -> bool:
        return(k==pygame.K_RETURN or k == pygame.K_UP or k == pygame.K_DOWN or k == pygame.K_LEFT or k == pygame.K_RIGHT)

if __name__ == '__main__':

    cNum = rNum = 0
    obstacles_list = []
    IC_list = []
    robot_pos = ()
    pe = 0
    
    with open('config.txt', 'r') as f:
        lineNumber = 0

        for line in f:
            # get all number of each line
            numbers = re.findall(r"\d+", line)
            for i in range(len(numbers)):
                numbers[i] = int(numbers[i])
                
            lineNumber += 1

            if 1 == lineNumber:
                # this is the column number and row number
                cNum = numbers[0]
                rNum = numbers[1]
            elif 2 == lineNumber:
                # this are obstacles
                for i in range(0, len(numbers), 2):
                    obstacles_list.append((numbers[i], numbers[i+1]))
            elif 3 == lineNumber:
                for i in range(0, len(numbers), 2):
                    IC_list.append((numbers[i], numbers[i+1]))
            elif 4 == lineNumber:
                robot_pos = (numbers[0], numbers[1])
            else:
                pe = float(line)

    # TODO explan more 
    gridworld = Gridworld(cNum, rNum, obstacles_list, IC_list, robot_pos, pe)
    simulator = Simulator(gridworld, cNum*GRID_SIZE, rNum*GRID_SIZE)
    simulator.loop()
            
