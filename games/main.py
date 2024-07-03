import sys
import pygame
from pygame.locals import *
from game import Game
from constants import *

def main():
    pygame.display.init()
    pygame.font.init()
    pygame.joystick.init()
    pygame.mixer.init()

    playSurface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Snake Game')
    image = pygame.image.load('images/game.ico')
    pygame.display.set_icon(image)
    p1_auto = True
    p2_auto = True

    game = Game(playSurface, p1_auto, p2_auto)
    game.run()

if __name__ == "__main__":
    main()