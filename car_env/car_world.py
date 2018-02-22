import pygame, random
#Let's import the Car Class
from car_agent import Car

class Safe_Stop(object):
    def __init__(self):
        self.GREEN = (20, 255, 140)
        self.DARK_GREEN = (0,100,0)
        self.GREY = (210, 210 ,210)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.SCREENWIDTH=180
        self.SCREENHEIGHT=600
        self.size = (self.SCREENWIDTH, self.SCREENHEIGHT)
        self.speed = 0
        self.screen = 0
        self.all_coming_cars = []
        self.all_sprites_list = []
        self.playerCar = 0
    def make_sprites(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Safe Stopping")
        self.playerCar = Car(self.DARK_GREEN, 60, 80, 5, 5)
        self.playerCar.rect.x = 60
        self.playerCar.rect.y = self.SCREENHEIGHT - 100
        car1 = Car(self.RED, 60, 80, 5, 5)
        car1.rect.x = 60
        car1.rect.y = 100
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.playerCar)
        self.all_sprites_list.add(car1)
        self.all_coming_cars = pygame.sprite.Group()
        self.all_coming_cars.add(car1)
    def rollout(self,random_stop_prob):
        self.make_sprites()
        carryOn = True
        clock=pygame.time.Clock()
        while carryOn:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    carryOn=False

            for car in self.all_sprites_list:
                s = random.randint(-5,5)
                car.accelerate(s)

            car_collision_list = pygame.sprite.spritecollide(self.playerCar,self.all_coming_cars,False)
            for car in car_collision_list:
                print("Car crash!")
                #End Of Game
                carryOn=False
            self.all_sprites_list.update()

            #Drawing on Screen
            self.screen.fill(self.GREEN)
            #Draw The Road
            pygame.draw.rect(self.screen, self.GREY, [40,0, 100,self.SCREENHEIGHT])
            #Draw Line painting on the road
            #pygame.draw.line(self.screen, self.WHITE, [140,0],[140,self.SCREENHEIGHT],5)


            #Now let's draw all the sprites in one go. (For now we only have 1 sprite!)
            self.all_sprites_list.draw(self.screen)

            #Refresh Screen
            pygame.display.flip()

            #Number of frames per secong e.g. 60
            clock.tick(60)
            #self.screen.blit()
        #Reset Sprites and speed before next rollout
        self.all_coming_cars = []
        self.all_sprites_list = []
        self.playerCar = 0
        self.speed = 0
        pygame.quit()
