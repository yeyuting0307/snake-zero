import pygame

WIDTH, HEIGHT = 640, 480
WIDTH_GRID = [w * 20 for w in range(1, WIDTH // 20 - 1)]
HEIGHT_GRID = [h * 20 for h in range(1, HEIGHT // 20 - 1)]

RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)
YELLOW = pygame.Color(255, 255, 0)
SKYBLUE = pygame.Color(135, 206, 235)
PURPLE = pygame.Color(102, 51, 153)
ORANGE = pygame.Color(255, 123, 0)
GOLD = pygame.Color(255, 191, 0)
TIFFANY = pygame.Color(129, 216, 208)
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
GREY = pygame.Color(150, 150, 150)
LIGHTGREY = pygame.Color(220, 220, 220)

FONT_PATH = "fonts/PingFang.ttc"