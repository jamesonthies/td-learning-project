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
        self.test_run_count = 1
        self.epsilon = 0.1
        self.test_epsilon = 0.05
        self.a = 0.1
        self.n = 64
        self.discount = 1
        self.max_steps = 10000
        self.Q = np.random.sample([2,3,3,3,3,3,3,9])*100
        self.state = []
        self.next_state = []
        self.state_path = [] #some initial state
        self.action = 4
        self.action_path = []
        self.reward_path = []
        self.T = float('inf')
        self.show = 1

    def on_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode([640, 400], pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.running = True
        
    def on_event(self, event):
        #up 273, down 274, right 275, left 276
        if event.type == pygame.QUIT:
            self.running = False

    def loop(self):
        #render basic game
        
        #learning
        if(self.timer == 0):
            self.render_basics() if self.show == 1 else self.norender_basics()
            self.car.update_state()
            self.state = self.car.state
            self.state_path.append(self.car.state)
            self.action = self.choose_action()
            self.action_path.append(self.action)
            self.act()
            self.timer += 1
        else:
            if(self.timer < self.T):
                terminal = (self.timer > self.max_steps or self.check_collision())
                self.render_basics() if self.show == 1 else self.norender_basics()
                self.car.update_state()
                self.next_state = self.car.state
                reward = self.current_reward()
                #terminal
                if(terminal):
                    self.T = self.timer
                else:
                    self.action = self.choose_action(1)
                self.action_path.append(self.action)
                self.state_path.append(self.car.state)
                self.reward_path.append(reward)

            toa = self.timer-self.n
            if(toa >= 0):
                gain = 0.0
                for t in range(toa, min(self.T, toa+self.n)):
                    gain += (self.discount**(t-toa-1))*self.reward_path[t]

                if(toa + self.n < self.T):
                    q_index = copy.copy(self.state_path[toa+self.n])
                    q_index.append(self.action_path[toa+self.n])
                    gain += (self.discount**(self.n))*self.Q[tuple(q_index)]
                #print('----')
                #print(len(self.state_path))
                #print(len(self.action_path))
                q_update = copy.copy(self.state_path[toa])
                q_update.append(self.action_path[toa])
                if(self.reward_path[toa] != -500):
                    self.Q[tuple(q_update)] += self.a*(gain-self.Q[tuple(q_update)])

            if(toa == self.T-1):
                print('Run: %d' % self.run_count)
                self.run_count+=1
                self.reset()
            else:
                self.timer += 1     
            self.state = self.next_state
            self.act()  

    def test_loop(self):
        if(self.timer == 0):
            self.render_basics() if self.show == 1 else self.norender_basics()
            self.car.update_state()
            self.state = self.car.state
            self.action = self.choose_action(test_greedy=1)
            self.act()
            self.timer += 1
            return 0
        else:
            terminal = (self.timer > self.max_steps or self.check_collision())
            self.render_basics() if self.show == 1 else self.norender_basics()
            self.car.update_state()
            self.next_state = self.car.state
            self.action = self.choose_action()
            reward = self.current_reward()          
            if(terminal):
                print('Test Run: %d' % self.test_run_count)
                self.test_run_count+=1
                self.reset()
            else:
                self.timer += 1     
                self.state = self.next_state
                self.act()  
            return reward


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
        self.show = 0
        run_lim = self.run_count+num_runs
        while(self.run_count<run_lim):
            self.loop()
        self.show = 1

    def test_ws(self,num_runs):
        self.test_run_count = 1
        reward_total = 0
        if self.on_init() == False:
            self.running = False
        while( self.running and self.test_run_count <= num_runs):
            for event in pygame.event.get():
                self.on_event(event)
            r = self.test_loop()
            reward_total += r
            pygame.display.update()
        return reward_total/num_runs


    def test(self,num_runs):
        self.show = 0
        temp = copy.copy(self.epsilon)
        self.epsilon = self.test_epsilon
        self.test_run_count = 1
        reward_total = 0
        while(self.test_run_count <= num_runs):
            r = self.test_loop()
            reward_total+=r
        self.epsilon = temp
        self.show = 1
        return reward_total/num_runs



    def reset(self):
        self.car = Car()
        self.checkpoints = Checkpoints()
        self.timer = 0
        self.state = []
        self.next_state = []
        self.state_path = [] #some initial state
        self.action = 4
        self.action_path = []
        self.reward_path = []
        self.T = float('inf')

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
            reward = 0
        else:
            self.car.color = [0,0,225]
            reward = 0
        if(self.checkpoints.check_pass(self.car.corners)):
            reward = reward+10
        return reward

    def check_collision(self):
        if(on_track(self.car.corners[0]) and on_track(self.car.corners[1]) and on_track(self.car.corners[2]) and on_track(self.car.corners[3])):
            return False
        else:
            return True

    def choose_action(self, next=0,test_greedy=0):
        comp = self.epsilon if test_greedy == 0 else self.test_epsilon
        if(np.random.sample(1) < comp):
            action = np.random.randint(0,9)
        else:
            i = self.state if next == 0 else self.next_state
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
        self.velocity_step = 0.8
        self.velocity_limit = 0.8
        self.orientation = 270
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
    def __init__(self):
        self.checkpoints = [(50,166,60,10),(50,110,60,10),
        (110,50,10,60),(212,50,10,60),(315,50,10,60),(418,50,10,60),(520,50,10,60),\
        (530,110,60,10),(530,166,60,10),(530,223,60,10),(530,280,60,10),\
        (520,290,10,60),(465,290,10,60),(410,290,10,60),\
        (350,280,60,10),(350,230,60,10),\
        (340,170,10,60),(290,170,10,60),\
        (230,230,60,10),(230,280,60,10),\
        (220,290,10,60),(165,290,10,60),(110,290,10,60),\
        (50,280,60,10),(50,223,60,10)]
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
    r0 = model.test_ws(100)
    model.train(50)
    r100 = model.test_ws(100)
    model.train(50)
    r500 = model.test_ws(100)
    

    print(r0)
    print(r100)
    print(r500)