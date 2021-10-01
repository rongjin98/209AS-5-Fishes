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

# this simulator will let the robot running on the gridworld
class Simulator:
    def __init__(self, gridWorld:Gridworld, width, height) -> None:
        pygame.init()
        pygame.display.set_caption("Simulator")
        self.screen = pygame.display.set_mode((width, height))
        self.gridWorld = gridWorld
        self.robot = Robot()
        
    def loop(self) -> None:
        while True:
            self.draw_map()

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            decision = self.robot.makeDecision()
            self.gridWorld.takeAction(decision, self.gridWorld.pe)
            self.gridWorld.updateMap(0)
            # this is used to control the simulate speed
            pygame.time.wait(50)
            pygame.display.update()

    def draw_map(self) -> None:
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
                px = i*GRID_SIZE
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
                pe = numbers[0]

    # TODO explan more 
    gridworld = Gridworld(cNum, rNum, obstacles_list, IC_list, robot_pos, pe)
    simulator = Simulator(gridworld, cNum*GRID_SIZE, rNum*GRID_SIZE)
    simulator.loop()
            
