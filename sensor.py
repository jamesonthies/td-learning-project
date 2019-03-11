import math
import pygame
#from pygame.locals import *
from helpers import detect_h, detect_v

#global track variables
track1 = [[(50,50),(590,50)],[(50,50),(50,350)],[(590,50),(590,350)],[(50,350),(290,350)], \
        [(350,350),(590,350)],[(110,110), (530,110)],[(110,110),(110,290)],[(110,290),(230,290)], \
        [(530,110),(530,290)],[(410,290),(530,290)],[(230,170),(410,170)],[(230,170),(230,290)], \
        [(410,170),(410,290)],[(290,230),(350,230)],[(350,230),(350,350)],[(290,230),(290,350)]]

track2 = [[(50,50),(590,50)],[(50,50),(50,230)],[(590,50),(590,350)], \
        [(410,350),(590,350)],[(110,110), (530,110)],[(110,110),(110,170)], \
        [(530,110),(530,290)],[(470,290),(530,290)],[(110,170),(470,170)], \
        [(470,170),(470,290)],[(50,230),(410,230)],[(410,230),(410,350)]]


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

        if(min_distance < 25):
            self.state = 0
            self.color = [225,0,0]
        elif(min_distance < 50):
            self.state = 1
            self.color = [225,128,0]
        else:
            self.state = 2
            self.color = [0,255,0]
            
        self.distance = min_distance           

    def render(self, screen):
        pygame.draw.line(screen, self.color, (self.x, self.y), (self.x+self.distance*math.cos(self.orientation*.0174533),self.y+self.distance*math.sin(self.orientation*.0174533)),3)
       