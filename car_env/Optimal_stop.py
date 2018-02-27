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
        self.GREEN = (20, 255, 140)
        self.BLUE = (0,0,255)
        self.DARK_GREEN = (0,100,0)
        self.GREY = (210, 210 ,210)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.SCREENWIDTH=220
        self.SCREENHEIGHT=600
        self.size = (self.SCREENWIDTH, self.SCREENHEIGHT)
        self.speed = 0
        self.screen = 0
        self.start_x_agent = 80
        self.start_y_agent = self.SCREENHEIGHT - 100
        self.start_x_driver = 80
        self.start_y_driver = 100
        self.all_coming_cars = []
        self.all_sprites_list = []
        self.playerCar = 0
        self.max_position = 10**5
        self.min_position = -10**5
        self.max_distance = 10**12
        self.min_distance = 0
        self.max_speed = 20
        self.min_speed = -20
        self.max_acceleration = 5
        self.min_acceleration = -3
        self.goal_position = 5*10**4
        self.low = np.array([self.min_position,self.min_distance, self.min_speed])
        self.high = np.array([self.max_position,self.max_distance, self.max_speed])
        self.action_space = spaces.Box(low=self.min_acceleration, high=self.max_acceleration, shape=(1,))
        self.observation_space = spaces.Box(low=self.low, high=self.high)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        position, distance, speed = self.state
        speed += action
        speed = np.clip(speed, self.min_speed, self.max_speed)
        position += speed
        position = np.clip(position)

    # makes the car sprites
    def make_sprites(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Safe Stopping")
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
    def viewer(self,random_stop_prob):
        self.make_sprites()
        background = pygame.image.load('background2.jpeg')
        w , h = background.get_size()
        carryOn = True
        clock=pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.size)
        y0 = 0
        y1 = 0
        while carryOn:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    carryOn=False
            # Scroll the background to make it seem
            # as if the blue car is moving
            self.speed = np.clip(self.speed+random.randint(0,2), -self.max_speed, self.max_speed)
            y1 = (y1 + self.speed) % h
            self.screen.blit(background,(0,-(h-y1)))
            self.screen.blit(background,(0,y1))
            # Move the red car
            for car in self.all_coming_cars:
                s = random.randint(0,5)
                car.accelerate(s, self.speed)

            # Check for collisions
            car_collision_list = pygame.sprite.spritecollide(self.playerCar,self.all_coming_cars,False)
            for car in car_collision_list:
                print("Car crash!")
                #End Of Game
                carryOn=False
            self.all_sprites_list.update()

            #Draw Line painting on the road
            #pygame.draw.line(self.screen, self.WHITE, [140,0],[140,self.SCREENHEIGHT],5)


            #Now let's draw all the sprites in one go. (For now we only have 1 sprite!)
            self.all_sprites_list.draw(self.screen)

            #Refresh Screen
            pygame.display.flip()
            #Number of frames per secong e.g. 60
            clock.tick(60)
            y0 = y1
            #self.screen.blit()
        #Reset Sprites and speed before next rollout
        self.all_coming_cars = []
        self.all_sprites_list = []
        self.playerCar = 0
        self.speed = 0
        pygame.quit()
