import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pygame, random
#Let's import the Car Class
from car_agent import *

class Optimal_Stop(gym.Env):


    def __init__(self):
        self.max_position = 10**4
        self.min_position = -10**2
        self.max_distance = 10**12
        self.min_distance = 0
        self.max_speed = 20
        self.min_speed = -20
        self.max_acceleration = 5
        self.min_acceleration = -3
        self.goal_position = 5*10**2
        self.low = np.array([self.min_position,self.min_distance, self.min_speed])
        self.high = np.array([self.max_position,self.max_distance, self.max_speed])
        self.action_space = spaces.Box(low=self.min_acceleration, high=self.max_acceleration, shape=(1,))
        self.observation_space = spaces.Box(low=self.low, high=self.high)
        self.stop_prob = 0.05
        self.random_stop = bool(np.random.uniform() < self.stop_prob)
        if self.random_stop:
            self.stop_position = np.random.uniform(100,4*10**3)
        self.seed()
        self.reset()
        self.rand_stop()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        position, distance, speed = self.state
        speed += action
        speed = np.clip(speed, self.min_speed, self.max_speed)
        position += speed
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and speed<0): speed = 0
        done = bool(position >= self.goal_position)
        reward = -1.0
        self.driver_speed += np.random.normal(0,0.05)
        self.driver_position += self.driver_speed
        distance = (self.driver_position) - position
        crash = bool(distance <= 0)
        if crash:
            print('Car Crash! from step')
            reward = -5000
            done = True
        if reward == -10*3:
            done = True
        self.state = (position, distance, speed)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([180, 240, 15])
        self.driver_speed = 15
        self.driver_position = 420
        #Reset Sprites and speed before next rollout
        self.all_coming_cars = []
        self.all_sprites_list = []
        pygame.quit()
        return np.array(self.state)

    def rand_stop(self):
        if self.random_stop:
            if self.driver_position > self.stop_position:
                self.driver_position = self.stop_position

    # makes the car sprites
    def make_sprites(self):
        self.playerCar = Car(self.BLUE, 60, 80, 10)
        self.playerCar.rect.x = self.start_x_agent
        self.playerCar.rect.y = self.start_y_agent
        car1 = Car(self.RED, 60, 80, 20)
        car1.rect.x = self.start_x_driver
        car1.rect.y = self.start_y_driver
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.playerCar)
        self.all_sprites_list.add(car1)
        self.all_coming_cars = pygame.sprite.Group()
        self.all_coming_cars.add(car1)
    # simulate an episode of optimal stopping
    # stop_prob the probability of random stop
    def open_pygame(self):
        pygame.init()
        pygame.display.set_caption("Safe Stopping")
        self.GREEN = (20, 255, 140)
        self.BLUE = (0,0,255)
        self.DARK_GREEN = (0,100,0)
        self.GREY = (210, 210 ,210)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        SCREENWIDTH=220
        SCREENHEIGHT=600
        size = (SCREENWIDTH, SCREENHEIGHT)
        self.start_x_agent = 80
        self.start_y_agent = SCREENHEIGHT - 180
        self.start_x_driver = 80
        self.start_y_driver = 100
        self.make_sprites()
        self.background = pygame.image.load('background2.jpeg')
        w , self.h = self.background.get_size()
        self.clock=pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)
        self.y0 = 0
        self.y1 = 0

    def render(self):
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                carryOn=False
        # Scroll the background to make it seem
        # as if the blue car is moving
        self.y1 = (self.y1 + self.state[2]) % self.h
        self.screen.blit(self.background,(0,-(self.h-self.y1)))
        self.screen.blit(self.background,(0,self.y1))
        # Move the red car
        # print('current agent speed is', self.state[2])
        # print('current driver speed is', self.driver_speed)
        for car in self.all_coming_cars:
            car.accelerate(self.driver_speed, self.state[2])

        # Check for collisions
        car_collision_list = pygame.sprite.spritecollide(self.playerCar,self.all_coming_cars,False)
        for car in car_collision_list:
            print("Car crash! from render")
            #End Of Game
        self.all_sprites_list.update()

        #Draw Line painting on the road
        #pygame.draw.line(self.screen, self.WHITE, [140,0],[140,self.SCREENHEIGHT],5)


        #Now let's draw all the sprites in one go. (For now we only have 1 sprite!)
        self.all_sprites_list.draw(self.screen)

        #Refresh Screen
        pygame.display.flip()
        #Number of frames per secong e.g. 60
        self.clock.tick(60)
        self.y0 = self.y1
