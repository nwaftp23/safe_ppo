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
        self.SCREENWIDTH=800
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
        self.playerCar = Car(self.DARK_GREEN, 60, 80, 70)
        self.playerCar.rect.x = 160
        self.playerCar.rect.y = self.SCREENHEIGHT - 100
        car1 = Car(self.RED, 60, 80, random.randint(50,100))
        car1.rect.x = 60
        car1.rect.y = -100
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
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_x:
                         self.playerCar.moveRight(10)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.playerCar.moveLeft(5)
            if keys[pygame.K_RIGHT]:
                self.playerCar.moveRight(5)
            if keys[pygame.K_UP]:
                self.speed += 0.05
            if keys[pygame.K_DOWN]:
                self.speed -= 0.05


            #Game Logic
            for car in self.all_coming_cars:
                car.moveForward(self.speed)
                if car.rect.y > self.SCREENHEIGHT:
                    car.changeSpeed(random.randint(50,100))
                    car.repaint(self.RED)
                    car.rect.y = -200
            car_collision_list = pygame.sprite.spritecollide(self.playerCar,self.all_coming_cars,False)
            for car in car_collision_list:
                print("Car crash!")
                #End Of Game
                carryOn=False
            self.all_sprites_list.update()

            #Drawing on Screen
            self.screen.fill(self.GREEN)
            #Draw The Road
            pygame.draw.rect(self.screen, self.GREY, [40,0, 600,self.SCREENHEIGHT])
            #Draw Line painting on the road
            pygame.draw.line(self.screen, self.WHITE, [140,0],[140,self.SCREENHEIGHT],5)


            #Now let's draw all the sprites in one go. (For now we only have 1 sprite!)
            self.all_sprites_list.draw(self.screen)

            #Refresh Screen
            pygame.display.flip()

            #Number of frames per secong e.g. 60
            clock.tick(60)
        #Reset Sprites and speed before next rollout
        self.all_coming_cars = []
        self.all_sprites_list = []
        self.playerCar = 0
        self.speed = 0
        pygame.quit()


"""
pygame.init()

GREEN = (20, 255, 140)
DARK_GREEN = (0,100,0)
GREY = (210, 210 ,210)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
BLUE = (100, 100, 255)

speed = 1
colorList = (RED, GREEN, PURPLE, YELLOW, CYAN, BLUE, DARK_GREEN)


SCREENWIDTH=800
SCREENHEIGHT=600

size = (SCREENWIDTH, SCREENHEIGHT)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Car Racing")

#This will be a list that will contain all the sprites we intend to use in our game.
all_sprites_list = pygame.sprite.Group()


playerCar = Car(DARK_GREEN, 60, 80, 700)
playerCar.rect.x = 160
playerCar.rect.y = SCREENHEIGHT - 100

car1 = Car(RED, 60, 80, random.randint(50,100))
car1.rect.x = 60
car1.rect.y = -100


# Add the car to the list of objects
all_sprites_list.add(playerCar)
all_sprites_list.add(car1)

all_coming_cars = pygame.sprite.Group()
all_coming_cars.add(car1)


#Allowing the user to close the window...
carryOn = True
clock=pygame.time.Clock()

while carryOn:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                carryOn=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_x:
                     playerCar.moveRight(10)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            playerCar.moveLeft(5)
        if keys[pygame.K_RIGHT]:
            playerCar.moveRight(5)
        if keys[pygame.K_UP]:
            speed += 0.05
        if keys[pygame.K_DOWN]:
            speed -= 0.05


        #Game Logic
        for car in all_coming_cars:
            car.moveForward(speed)
            if car.rect.y > SCREENHEIGHT:
                car.changeSpeed(random.randint(50,100))
                car.repaint(random.choice(colorList))
                car.rect.y = -200
        car_collision_list = pygame.sprite.spritecollide(playerCar,all_coming_cars,False)
        for car in car_collision_list:
            print("Car crash!")
            #End Of Game
            carryOn=False
        all_sprites_list.update()

        #Drawing on Screen
        screen.fill(GREEN)
        #Draw The Road
        pygame.draw.rect(screen, GREY, [40,0, 600,SCREENHEIGHT])
        #Draw Line painting on the road
        pygame.draw.line(screen, WHITE, [140,0],[140,SCREENHEIGHT],5)
        #Draw Line painting on the road
        pygame.draw.line(screen, WHITE, [240,0],[240,SCREENHEIGHT],5)
        #Draw Line painting on the road
        pygame.draw.line(screen, WHITE, [340,0],[340,SCREENHEIGHT],5)


        #Now let's draw all the sprites in one go. (For now we only have 1 sprite!)
        all_sprites_list.draw(screen)

        #Refresh Screen
        pygame.display.flip()

        #Number of frames per secong e.g. 60
        clock.tick(60)

pygame.quit()
"""
