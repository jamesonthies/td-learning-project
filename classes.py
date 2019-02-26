import math
import copy
import pygame
from pygame.locals import *
import numpy as np
from helpers import on_track,detect_h,detect_v


#global track variables
background_color = [20,20,30]
#track_color = [175,200,200]
track_color = [20,20,30]
track1 = [[(50,50),(590,50)],[(50,50),(50,350)],[(590,50),(590,350)],[(50,350),(290,350)], \
        [(350,350),(590,350)],[(110,110), (530,110)],[(110,110),(110,290)],[(110,290),(230,290)], \
        [(530,110),(530,290)],[(410,290),(530,290)],[(230,170),(410,170)],[(230,170),(230,290)], \
        [(410,170),(410,290)],[(290,230),(350,230)],[(350,230),(350,350)],[(290,230),(290,350)]]

track2 = [[(50,50),(590,50)],[(50,50),(50,230)],[(590,50),(590,350)], \
        [(350,350),(590,350)],[(110,110), (530,110)],[(110,110),(110,170)], \
        [(530,110),(530,290)],[(410,290),(530,290)],[(110,170),(410,170)], \
        [(410,170),(410,290)],[(50,230),(350,230)],[(350,230),(350,350)]]

class Car:
    def __init__(self, track=1):
        self.location_x = 560 if track == 2 else 80
        self.location_y = 200 if track == 2 else 200
        self.velocity = 0
        self.velocity_step = 0.8
        self.velocity_limit = 0.8
        self.orientation = 90 if track == 2 else 270
        self.orientation_change = 0
        self.orientation_step = 1
        self.orientation_change_limit = 1
        self.corners = []
        self.color = [0,255,0]
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


class Sensor:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.orientation = 0
        self.distance = 100
        self.max_distance = 1000
        self.color = [0,0,0]
        self.state = 0

    def detect(self, track):
        min_distance = self.max_distance
        track_list = track2 if track == 2 else track1

        for segment in track_list:
            if(segment[0][1] == segment[1][1]):
                temp = detect_h([self.x,self.y],self.orientation,segment)
                if(temp != -1):
                    if(self.orientation > 180):
                        pass
            elif(segment[0][0] == segment[1][0]):
                temp = detect_v([self.x,self.y],self.orientation,segment)
            else:
                temp = -1

            if(temp < min_distance and temp != -1):
                min_distance = temp

        '''
        #Sensor returns 0 or 1
        if(min_distance < 60):
            self.state = 0
            self.color = [255,0,0]
        else:
            self.state = 1
            self.color = [0,255,0]
        '''
        if(min_distance < 25):
            self.state = 0
            self.color = [225,0,0]
        elif(min_distance < 50):
            self.state = 1
            #self.color = [182,219,102]
            self.color = [225,128,0]
            #self.state = 0
            #self.color = [255,0,0]
        elif(min_distance< 75):
            self.state = 2
            #self.color = [125,190,102]
            self.color = [0,255,0]
            #self.state = 1
            #self.color = [0,255,0]
        elif(min_distance< 100):
            self.state = 2
            #self.color = [81,168,102]
            self.color =  [0,255,0]
            #self.state = 1
            #self.color = [0,255,0]
        else:
            self.state = 2
            self.color =  [0,255,0]
            #self.color = [182,219,102]
            #self.state = 1
            #self.color = [0,255,0]
        self.distance = min_distance           

    def render(self, screen):
        pygame.draw.line(screen, self.color, (self.x, self.y), (self.x+self.distance*math.cos(self.orientation*.0174533),self.y+self.distance*math.sin(self.orientation*.0174533)),3)
        

class Checkpoints:
    def __init__(self,track=1):
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
            (520,290,10,60),(465,290,10,60),(410,290,10,60),\
            (350,280,60,10), (350,230,60,10),\
            (340,170,10,60),(225,170,10,60),(110,170,10,60),\
            (50,160,60,10), (50,110,60,10),\
            (110,50,10,60),(212,50,10,60),(315,50,10,60),(418,50,10,60),(520,50,10,60),\
            (530,110,60,10),(530,166,60,10),]
        self.active = 0

    def render_all(self, screen):
        for checkpoint in self.checkpoints:
            pygame.draw.rect(screen, [0,255,0], checkpoint)

    def render_active(self, screen):
        pygame.draw.rect(screen, [255,255,0], self.checkpoints[self.active])        

    def check_pass(self,car):
        curr = self.checkpoints[self.active]
        for corner in car:
            if(corner[0] > curr[0] and corner[0] < curr[0]+curr[2] and corner[1] > curr[1] and corner[1] < curr[1]+curr[3]):
                self.active = (self.active+1)%(len(self.checkpoints))
                return True
            else:
                return False                
