import math
import copy
import time
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from helpers import on_track
from car import Car
from checkpoints import Checkpoints

#global track variables
background_color = [20,20,20]
track_color = [200,200,200]
#track_color = [200,200,30]

pos_reward = 10
neg_reward = -50

class Model:
    def __init__(self,n):
        self.running = True
        self.screen = None
        self.car = Car()
        self.checkpoints = Checkpoints()
        self.timer = 0
        self.run_count = 1
        self.test_run_count = 1
        self.epsilon = 0.01
        self.a = 0.01
        self.n = n
        self.discount = 1
        self.max_steps = 8000
        self.Q = np.random.sample([2,3,3,3,3,3,3,9])+10
        self.state = []
        self.next_state = []
        self.action = 0
        self.state_path = [] #some initial state
        self.action_path = []
        self.reward_path = []
        self.ep_rewards = []
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
            self.render_train() if self.show == 1 else self.norender_train()
            self.car.update_state()
            self.state = self.car.state
            self.state_path.append(self.car.state)
            self.action = self.choose_action()
            self.action_path.append(self.action)
            self.act()
            self.reward_path.append(0)
            self.timer += 1
        else:
            if(self.timer < self.T):
                terminal = (self.timer > self.max_steps or self.check_collision())
                self.render_train() if self.show == 1 else self.norender_train()
                self.car.update_state()
                self.next_state = self.car.state
                reward = self.current_reward()
                #terminal
                if(terminal):
                    self.T = self.timer
                else:
                    self.action = self.choose_action(1)
                    self.reward_path.append(reward)
                self.action_path.append(self.action)
                self.state_path.append(self.car.state)
                
            toa = self.timer-self.n
            if(toa >= 0):
                gain = 0.0
                for t in range(toa, min(self.T, toa+self.n)):
                    gain += (self.discount**(t-toa-1))*self.reward_path[t]

                if(toa + self.n < self.T):
                    q_index = copy.copy(self.state_path[toa+self.n])
                    q_index.append(self.action_path[toa+self.n])
                    gain += (self.discount**(self.n))*self.Q[tuple(q_index)]
                q_update = copy.copy(self.state_path[toa])
                q_update.append(self.action_path[toa])
                if(self.reward_path[toa] != neg_reward):
                    self.Q[tuple(q_update)] += self.a*(gain-self.Q[tuple(q_update)])

            if(toa == self.T-1):
                print(f'Run: {self.run_count}, res: {sum(self.reward_path)}')
                self.run_count+=1
                self.ep_rewards.append(sum(self.reward_path))
                self.reset()
            else:
                self.timer += 1     
            self.state = self.next_state
            self.act()  

    def test_loop(self, track):
        if(self.timer == 0):
            self.render_test(track) if self.show == 1 else self.norender_test(track)
            self.car.update_state()
            self.state = self.car.state
            self.state_path.append(self.car.state)
            self.action = self.choose_action()
            self.action_path.append(self.action)
            self.act()
            self.reward_path.append(0)
            self.timer += 1
            return 0
        else:
            if(self.timer < self.T):
                terminal = (self.timer > self.max_steps or self.check_collision(track))
                self.render_test(track) if self.show == 1 else self.norender_test(track)
                self.car.update_state()
                self.next_state = self.car.state
                reward = 0 if self.reward_path[len(self.reward_path)-1] == neg_reward else self.current_reward(track)
                #terminal
                if(terminal):
                    self.T = self.timer
                else:
                    self.action = self.choose_action(1)
                    self.reward_path.append(reward)
                self.action_path.append(self.action)
                self.state_path.append(self.car.state)
            else:
                reward = 0
            toa = self.timer-self.n
            

            if(toa == self.T-1):
                print(f'TEST: {self.test_run_count}, res: {sum(self.reward_path)}')
                self.test_run_count+=1
                self.reset(track)
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

    def train_wr(self, num_runs):
        self.reset()
        run_lim = self.run_count+num_runs
        if self.on_init() == False:
            self.running = False
        while( self.running and self.run_count < run_lim):
            for event in pygame.event.get():
                self.on_event(event)
            self.loop()
            pygame.display.update()
        self.clean()

    def train(self, num_runs):
        self.reset()
        self.show = 0
        run_lim = self.run_count+num_runs
        while(self.run_count<run_lim):
            self.loop()
        self.show = 1

    def test_wr(self,num_runs,track):
        self.reset(track)
        self.test_run_count = 1
        reward_total = 0
        if self.on_init() == False:
            self.running = False
        while( self.running and self.test_run_count <= num_runs):
            for event in pygame.event.get():
                self.on_event(event)
            r = self.test_loop(track)
            reward_total += r
            pygame.display.update()
        self.clean()
        return reward_total/num_runs


    def test(self,num_runs,track):
        self.reset(track)
        self.show = 0
        temp = copy.copy(self.epsilon)
        self.test_run_count = 1
        reward_total = 0
        while(self.test_run_count <= num_runs):
            r = self.test_loop(track)
            reward_total+=r
        self.epsilon = temp
        self.show = 1
        return reward_total/num_runs

    def reset(self,track=1):
        self.car = Car(track)
        self.checkpoints = Checkpoints(track)
        self.timer = 0
        self.state = []
        self.next_state = []
        self.state_path = [] #some initial state
        self.action = 4
        self.action_path = []
        self.reward_path = []
        self.T = float('inf')

    def render_train(self):
        #render the game elements
        self.screen.fill(background_color)
        self.render_track1()
        self.checkpoints.render(self.screen)
        self.car.move()
        self.car.render(self.screen)

    def norender_train(self):
        self.car.move()

    def render_test(self,track):
        #render the game elements
        self.screen.fill(background_color)
        if(track == 2):
            self.render_track2()
        else:
            self.render_track1()
        self.checkpoints.render(self.screen)
        self.car.move(track)
        self.car.render(self.screen)

    def norender_test(self,track):
        self.car.move(track)

    def render_track1(self):
        pygame.draw.rect(self.screen, track_color, [50,50,60,300])
        pygame.draw.rect(self.screen, track_color, [50,50,540,60])
        pygame.draw.rect(self.screen, track_color, [50,290,240,60])
        pygame.draw.rect(self.screen, track_color, [230,170,60,180])
        pygame.draw.rect(self.screen, track_color, [230,170,180,60])
        pygame.draw.rect(self.screen, track_color, [530,50,60,300])
        pygame.draw.rect(self.screen, track_color, [350,170,60,180])
        pygame.draw.rect(self.screen, track_color, [350,290,240,60])        

    def render_track2(self):
        pygame.draw.rect(self.screen, track_color, [50,50,540,60])
        pygame.draw.rect(self.screen, track_color, [530,50,60,300])
        pygame.draw.rect(self.screen, track_color, [410,170,60,180])
        pygame.draw.rect(self.screen, track_color, [410,290,180,60])
        pygame.draw.rect(self.screen, track_color, [50,170,420,60])
        pygame.draw.rect(self.screen, track_color, [50,50,60,180])

    #ADD Track
    def current_reward(self,track=1):
        reward = 0
        if(self.check_collision(track)):
            self.car.color = [255,0,0]
            reward = neg_reward
        else:
            self.car.color = [10,46,73]
            reward = 0
        if(self.checkpoints.check_pass(self.car.corners)):
            reward = reward+pos_reward
        return reward

    #ADD TRACK
    def check_collision(self,track=1):
        if(on_track(self.car.corners[0],track) and on_track(self.car.corners[1],track) and on_track(self.car.corners[2],track) and on_track(self.car.corners[3],track)):
            return False
        else:
            return True

    def choose_action(self, next=0):
        if(np.random.sample(1) < self.epsilon):
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


if __name__ == "__main__" :
    n = 64
    num_runs = 10
    test_length = 10
    train_jumps = [10,40,50,100,100,100,100,100,100,100,100,100]


    start_t = time.time()
    res = np.zeros(sum(train_jumps))
    test1 =  np.zeros(len(train_jumps)+1)
    test2 =  np.zeros(len(train_jumps)+1)
    
    for i in range(num_runs):
        #
        model64 = Model(n)
        r = model64.test(test_length,1)
        print(f'{n}.{i}.Result: {r}')
        test1[0] += r
        r = model64.test(test_length,2)
        print(f'{n}.{i}.Result: {r}')
        test2[0] += r
        for i2 in range(len(train_jumps)):    
            model64.train(train_jumps[i2])#10
            r = model64.test_wr(test_length,1)
            print(f'{n}.{i}.Result: {r}')
            test1[i2+1] += r
            r = model64.test_wr(test_length,2)
            print(f'{n}.{i}.Result: {r}')
            test2[i2+1] += r

        res += model64.ep_rewards
        print(f'Iteration {i} Complete')

    #Displaying information
    res /= num_runs
    plt.plot(res)
    plt.title('n = 64')
    plt.show()
    print('>>>Results: n = 64')

    print(test1) 
    print(test2)
    test1 /= num_runs
    test2 /= num_runs
    print(test1)
    print(test2)
    end_t = time.time()
    print(end_t-start_t)