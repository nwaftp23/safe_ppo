import pygame, random
#Let's import the Car Class
from car_agent import *

class Safe_Stop(object):
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
        self.all_coming_cars = []
        self.all_sprites_list = []
        self.playerCar = 0
        self.max_speed = 20

    # makes the car sprites
    def make_sprites(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Safe Stopping")
        self.playerCar = Car(self.BLUE, 60, 80, 10)
        self.playerCar.rect.x = 80
        self.playerCar.rect.y = self.SCREENHEIGHT - 100
        car1 = Car(self.RED, 60, 80, 20)
        car1.rect.x = 80
        car1.rect.y = 100
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.playerCar)
        self.all_sprites_list.add(car1)
        self.all_coming_cars = pygame.sprite.Group()
        self.all_coming_cars.add(car1)
    # Makes and saves background
    # then I can use blit to scroll
    # Only use this function to make a new make background
    # Or to add a background to the directory
    def make_background(self,filename):
        pygame.init()
        self.screen = pygame.display.set_mode((220,1040))
        self.screen.fill(self.GREEN)
        #Draw The Road
        pygame.draw.rect(self.screen, self.GREY, [60,0, 100,1040])
        pygame.draw.rect(self.screen, self.DARK_GREEN, [15,40, 30,40])
        pygame.draw.rect(self.screen, self.DARK_GREEN, [175,520, 30,40])
        pygame.image.save(self.screen, filename)
    # simulate an episode of optimal stopping
    # stop_prob the probability of random stop
    def rollout(self,random_stop_prob):
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
