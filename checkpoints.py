#checkpoints.py
#Code availale at https://github.com/jamesonthies/td-learning-project

#Jameson Thies
#Kourosh Vali
#Yanda Chen

import pygame

#A checkpoint is just a rectangle. 
#The checkpoints is just a list of rectangels
#This object defines a list of ceckpoints, and keeps track of which one is active
#It also checks to see if the car is currently passing a checkpoint
class Checkpoints:
    def __init__(self,track=1):
        #Checkpoint list based on track
        if(track != 2):
            self.checkpoints = [(50,166,60,10),(50,110,60,10),
            (110,50,10,60),(212,50,10,60),(315,50,10,60),(418,50,10,60),(520,50,10,60),\
            (530,110,60,10),(530,166,60,10),(530,223,60,10),(530,280,60,10),\
            (520,290,10,60),(465,290,10,60),(410,290,10,60),\
            (350,280,60,10),(350,230,60,10),\
            (340,170,10,60),(290,170,10,60),\
            (230,230,60,10),(230,280,60,10),\
            (220,290,10,60),(165,290,10,60),(110,290,10,60),\
            (50,280,60,10),(50,223,60,10)]
        else:      
            self.checkpoints = [(530,223,60,10),(530,280,60,10),\
            (520,290,10,60),(470,290,10,60),\
            (410,280,60,10), (410,230,60,10),\
            (400,170,10,60),(303,170,10,60),(207,170,10,60),(110,170,10,60),\
            (50,160,60,10), (50,110,60,10),\
            (110,50,10,60),(212,50,10,60),(315,50,10,60),(418,50,10,60),(520,50,10,60),\
            (530,110,60,10),(530,166,60,10)]
        #Variable to keep track of which checkpoint is active
        self.active = 0

    #Renders checkpoints to pygame screen
    def render(self, screen):
        for checkpoint in self.checkpoints:
            if(checkpoint != self.checkpoints[self.active]):
                pygame.draw.rect(screen, [150,150,0], checkpoint)
        pygame.draw.rect(screen, [255,255,0], self.checkpoints[self.active])          

    #looks to see if a checkpoint is being passed,
    #When a checkpoint is passed, it is deactivated and the next one is activated
    def check_pass(self,car):
        curr = self.checkpoints[self.active]
        for corner in car:
            if(corner[0] > curr[0] and corner[0] < curr[0]+curr[2] and corner[1] > curr[1] and corner[1] < curr[1]+curr[3]):
                self.active = (self.active+1)%(len(self.checkpoints))
                return True
            else:
                return False                
