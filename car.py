import math
import pygame
from sensor import Sensor

#from pygame.locals import *

class Car:
    def __init__(self, track=1):
        self.location_x = 560 if track == 2 else 80
        self.location_y = 200 if track == 2 else 200
        self.velocity = 0
        self.velocity_step = 1
        self.velocity_limit = 1 
        self.orientation = 90 if track == 2 else 270
        self.orientation_change = 0
        self.orientation_step = 1
        self.orientation_change_limit = 1
        self.corners = []
        self.color = [10,46,73]
        self.size = 15
        self.sensor1 = Sensor()
        self.sensor2 = Sensor()
        self.sensor3 = Sensor()
        self.sensor4 = Sensor()
        self.sensor5 = Sensor()
        self.state = []

    def move(self,track=1):
        self.orientation = (self.orientation + self.orientation_change)%360
        self.location_y += math.sin(self.orientation*0.0174533)*self.velocity
        self.location_x += math.cos(self.orientation*0.0174533)*self.velocity
        self.sense(track)
        self.update_state()
        point_set = [
        [self.location_x+math.cos((self.orientation+340)*.0174533)*self.size, self.location_y+math.sin((self.orientation+340)*0.0174533)*self.size],\
        [self.location_x+math.cos((self.orientation+20)*0.0174533)*self.size, self.location_y+math.sin((self.orientation+20)*0.0174533)*self.size],\
        [self.location_x+math.cos((self.orientation+160)*0.0174533)*self.size, self.location_y+math.sin((self.orientation+160)*0.0174533)*self.size],\
        [self.location_x+math.cos((self.orientation+200)*0.0174533)*self.size, self.location_y+math.sin((self.orientation+200)*0.0174533)*self.size]]
        self.corners = point_set

    def sense(self,track):
        self.sensor1.x = self.location_x
        self.sensor2.x = self.location_x
        self.sensor3.x = self.location_x
        self.sensor4.x = self.location_x
        self.sensor5.x = self.location_x
        
        self.sensor1.y = self.location_y
        self.sensor2.y = self.location_y
        self.sensor3.y = self.location_y
        self.sensor4.y = self.location_y
        self.sensor5.y = self.location_y
        
        self.sensor1.orientation = (self.orientation-90)%360
        self.sensor2.orientation = (self.orientation-45)%360
        self.sensor3.orientation = (self.orientation)%360
        self.sensor4.orientation = (self.orientation+45)%360
        self.sensor5.orientation = (self.orientation+90)%360
        
        self.sensor1.detect(track)
        self.sensor2.detect(track)
        self.sensor3.detect(track)
        self.sensor4.detect(track)
        self.sensor5.detect(track)

    def pedal(self, direction):
        if(direction == 0):
            self.velocity -= self.velocity_step
        if(direction == 2):
            self.velocity += self.velocity_step
        self.velocity = min(self.velocity, self.velocity_limit)
        self.velocity = max(self.velocity, 0)

    def steer(self, direction):
        if(direction == 0):
            #left
            self.orientation_change -= self.orientation_step 
        elif(direction == 2):
            #right
            self.orientation_change += self.orientation_step
        self.orientation_change = min(self.orientation_change, self.orientation_change_limit)
        self.orientation_change = max(self.orientation_change, -1*self.orientation_change_limit)

    def update_state(self):
        self.state = [round(self.velocity/self.velocity_step),round(self.orientation_change/self.orientation_step)+1,self.sensor1.state,self.sensor2.state,self.sensor3.state,self.sensor4.state,self.sensor5.state]

    def render(self, screen):
        self.sensor1.render(screen)
        self.sensor2.render(screen)
        self.sensor3.render(screen)
        self.sensor4.render(screen)
        self.sensor5.render(screen)
        pygame.draw.polygon(screen,tuple(self.color), self.corners)
