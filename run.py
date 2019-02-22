import math
import pygame
from pygame.locals import *
import numpy as np
from helpers import on_track,detect_h,detect_v

#global track variables
background_color = [20,20,30]
track_color = [175,200,200]
track = [[(50,50),(590,50)],[(50,50),(50,350)],[(590,50),(590,350)],[(50,350),(290,350)], \
        [(350,350),(590,350)],[(110,110), (530,110)],[(110,110),(110,290)],[(110,290),(230,290)], \
        [(530,110),(530,290)],[(410,290),(530,290)],[(230,170),(410,170)],[(230,170),(230,290)], \
        [(410,170),(410,290)],[(290,230),(350,230)],[(350,230),(350,350)],[(290,230),(290,350)]]

class Model:
    def __init__(self):
        self.running = True
        self.screen = None
        self.car = Car()
        self.checkpoints = Checkpoints()
        self.timer = 0
        self.run_count = 1
        self.epsilon = 0.3
        self.step_size = 0.9
        self.discount = 0.9
        self.max_steps = 10000
        self.Q = np.random.sample([5,5,5,5,5,5,5,9])*10
        #self.Q = np.zeros([5,5,2,2,2,2,2,9])
        self.state = [] #some initial state
        self.next_state = []
        self.action = 4

    def on_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode([640, 400], pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.running = True
        
    def on_event(self, event):
        #up 273, down 274, right 275, left 276
        if event.type == pygame.QUIT:
            self.running = False

    def loop(self, training=0):
        #render basic game
        if(training != 1):
            self.render_basics()
        else:
            self.norender_basics()

        self.timer += 1        
        if(self.timer == 1):
            self.car.update_state()
            self.state = self.car.state
            self.action = self.choose_action()
            self.act()
        else:
            self.car.update_state()
            self.next_state = self.car.state
            reward = self.current_reward()
            sa_array = np.concatenate((self.state,[self.action]),axis=None)
            nsa_array = np.concatenate((self.next_state, [np.argmax(tuple(self.state))]),axis=None)
            qnsa = self.Q[tuple(nsa_array)]
            self.Q[tuple(sa_array)] += self.step_size*(reward+self.discount*qnsa-self.Q[tuple(sa_array)])
            self.state = self.next_state
            self.action = self.choose_action()
            self.act()
        #if(self.timer > 5000 or self.check_collision()):
        #    self.reset()
        if(self.timer > self.max_steps or self.check_collision()):
            print('Run: %d' % self.run_count)
            self.run_count+=1
            self.reset()
        #if(self.check_collision()):
        #    print('Run: %d' % self.run_count)
        #    self.run_count+=1
        #    self.reset()

    def clean(self):
        pygame.quit()
 
    def run(self):
        if self.on_init() == False:
            self.running = False
        while( self.running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.loop()
            pygame.display.update()
        self.clean()

    def train(self, num_runs):
        while(self.run_count<num_runs):
            self.loop(1)

    def reset(self):
        self.car = Car()
        self.checkpoints = Checkpoints()
        self.timer = 0

    def render_basics(self):
        #render the game elements
        self.screen.fill(background_color)
        self.render_track()
        self.checkpoints.render_active(self.screen)
        self.car.move()
        self.car.render(self.screen)

    def norender_basics(self):
        self.car.move()

    def render_track(self):
        pygame.draw.rect(self.screen, track_color, [50,50,60,300])
        pygame.draw.rect(self.screen, track_color, [50,50,540,60])
        pygame.draw.rect(self.screen, track_color, [50,290,240,60])
        pygame.draw.rect(self.screen, track_color, [230,170,60,180])
        pygame.draw.rect(self.screen, track_color, [230,170,180,60])
        pygame.draw.rect(self.screen, track_color, [530,50,60,300])
        pygame.draw.rect(self.screen, track_color, [350,170,60,180])
        pygame.draw.rect(self.screen, track_color, [350,290,240,60])        

    def current_reward(self):
        reward = 0
        if(self.check_collision()):
            self.car.color = [255,0,0]
            reward = -500
        else:
            self.car.color = [0,0,225]
            reward = -1
        if(self.checkpoints.check_pass(self.car.corners)):
            reward = reward+1001
        return reward


    def check_collision(self):
        if(on_track(self.car.corners[0]) and on_track(self.car.corners[1]) and on_track(self.car.corners[2]) and on_track(self.car.corners[3])):
            return False
        else:
            return True

    def choose_action(self):
        if(np.random.sample(1) < self.epsilon):
            action = np.random.randint(0,9)
        else:
            i = self.state
            action_choices = self.Q[i[0],i[1],i[2],i[3],i[4],i[5],i[6],:]
            action = np.argmax(action_choices)
        return action

    def act(self):
        accel = math.floor(self.action/3)
        turn = self.action%3
        self.car.pedal(accel)
        self.car.steer(turn)

class Car:
    def __init__(self):
        self.location_x = 80
        self.location_y = 200
        self.velocity = 0
        self.velocity_step = 0.1
        self.velocity_limit = 0.4
        self.orientation = 270
        self.orientation_change = 0
        self.orientation_step = 0.08
        self.orientation_change_limit = 0.16
        self.corners = []
        self.color = [0,255,0]
        self.size = 15
        self.sensor1 = Sensor()
        self.sensor2 = Sensor()
        self.sensor3 = Sensor()
        self.sensor4 = Sensor()
        self.sensor5 = Sensor()
        self.state = []

    def move(self):
        self.orientation = (self.orientation + self.orientation_change)%360
        self.location_y += math.sin(self.orientation*0.0174533)*self.velocity
        self.location_x += math.cos(self.orientation*0.0174533)*self.velocity
        self.sense()
        self.update_state()
        point_set = [
        [self.location_x+math.cos((self.orientation+340)*.0174533)*self.size, self.location_y+math.sin((self.orientation+340)*0.0174533)*self.size],\
        [self.location_x+math.cos((self.orientation+20)*0.0174533)*self.size, self.location_y+math.sin((self.orientation+20)*0.0174533)*self.size],\
        [self.location_x+math.cos((self.orientation+160)*0.0174533)*self.size, self.location_y+math.sin((self.orientation+160)*0.0174533)*self.size],\
        [self.location_x+math.cos((self.orientation+200)*0.0174533)*self.size, self.location_y+math.sin((self.orientation+200)*0.0174533)*self.size]]
        self.corners = point_set

    def sense(self):
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
        
        self.sensor1.detect()
        self.sensor2.detect()
        self.sensor3.detect()
        self.sensor4.detect()
        self.sensor5.detect()

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
        self.state = [round(self.velocity/self.velocity_step),round(self.orientation_change/self.orientation_step)+2,self.sensor1.state,self.sensor2.state,self.sensor3.state,self.sensor4.state,self.sensor5.state]

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

    def detect(self):
        min_distance = self.max_distance
        for segment in track:
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
            self.color = [255,0,0]
        elif(min_distance < 50):
            self.state = 1
            self.color = [255,128,0]
        elif(min_distance<75):
            self.state = 2
            self.color = [255,255,0]
        elif(min_distance<100):
            self.state = 3
            self.color = [48,255,207]
        else:
            self.state = 4
            self.color = [0,0,225]
        self.distance = min_distance           

    def render(self, screen):
        pygame.draw.line(screen, self.color, (self.x, self.y), (self.x+self.distance*math.cos(self.orientation*.0174533),self.y+self.distance*math.sin(self.orientation*.0174533)),3)
        

class Checkpoints:
    def __init__(self):
        self.checkpoints = [(50,150,60,10), (150,50,10,60),(315,50,10,60),(480,50,10,60),\
        (530,150,60,10),(530,250,60,10),(465,290,10,60),(350,255,60,10),(315,170,10,60),\
        (230,255,60,10),(165,290,10,60),(50,250,60,10)]
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

if __name__ == "__main__" :
    model = Model()
    model.train(10)
    model.run()        