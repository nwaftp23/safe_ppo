import pygame
WHITE = (255, 255, 255)

class Car(pygame.sprite.Sprite):
    #This class represents a car. It derives from the "Sprite" class in Pygame.

    def __init__(self, color, width, height, max_speed, max_acceleration):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Pass in the color of the car, and its x and y position, width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width, height])
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)

        #Initialise attributes of the car.
        self.width = width
        self.height = height
        self.color = color
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.speed = 0

        # Draw the car (a rectangle!)
        pygame.draw.rect(self.image, self.color, [0, 0, self.width, self.height])

        # Instead we could load a proper picture of a car...
        # self.image = pygame.image.load("car.png").convert_alpha()

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()

    def move_car(self):
        if abs(self.speed) < self.max_speed:
            self.rect.y += self.speed
        else:
            self.rect.y += self.max_speed

    def accelerate(self, speed):
        if speed < self.max_acceleration:
            self.speed += speed
        else:
            self.speed += self.max_acceleration
        self.move_car()

    def changeSpeed(self, speed):
        self.speed = speed

    def repaint(self, color):
        self.color = color
        pygame.draw.rect(self.image, self.color, [0, 0, self.width, self.height])
