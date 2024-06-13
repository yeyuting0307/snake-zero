import sys
import time
import random
import pygame
from pygame.locals import *
from constants import *
from utils import showText, gameStart, gameRule, gameOver

class Game:
    def __init__(self, playSurface):
        self.playSurface = playSurface
        self.fpsClock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snakePosition1 = [100, 100]
        self.snakeSegments1 = [[100, 100], [80, 100], [60, 100]]
        
        self.snakePosition2 = [200, 200]
        self.snakeSegments2 = [[200, 200], [220, 200], [240, 200]]
        
        self.candySpawned = 1
        self.candyNum = 1
        init_x = random.sample(WIDTH_GRID, self.candyNum)
        init_y = random.sample(HEIGHT_GRID, self.candyNum)
        self.candyPosition = [list(z) for z in zip(init_x, init_y)]
        
        
        self.direction1 = 'right'
        self.direction2 = 'left'
        
        self.changeDirection1 = self.direction1
        self.changeDirection2 = self.direction2
        
        self.score1 = 0
        self.score2 = 0
        
        self.foul1 = 0
        self.foul2 = 0
        
        self.snakeColor1 = YELLOW
        self.snakeColor2 = SKYBLUE
        
        self.scoreColor1 = GREY
        self.scoreColor2 = GREY
        
        self.timeColor = GREY
        
        self.border = False
        self.gameRestrictTime = 42
        self.gameSpeed = 16
        self.pauseTotalTime = 0
        self.running = True
        
    def run(self):
        holdStart = True
        while holdStart:
            holdStart = gameStart(self.playSurface)
        
        self.playSurface.fill(BLACK)
        ruleStart = True
        while ruleStart:
            ruleStart = gameRule(self.playSurface)
        
        restart = True
        while restart:
            self.reset()
            self.gameStartTime = pygame.time.get_ticks()
            
            while self.running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == KEYDOWN:
                        self.handle_keydown(event)
                        
                self.update_game()
                self.render_game()
                
                passTime = (pygame.time.get_ticks() - self.gameStartTime) / 1000 - self.pauseTotalTime
                if self.gameRestrictTime - passTime <= 0:
                    self.running = False
                    gameOver(self.playSurface, self.score1, self.score2)
                    
                self.fpsClock.tick(self.gameSpeed)
    
    def handle_keydown(self, event):
        if event.key == ord('d'):
            self.changeDirection1 = 'right'
        if event.key == ord('a'):
            self.changeDirection1 = 'left'
        if event.key == ord('w'):
            self.changeDirection1 = 'up'
        if event.key == ord('s'):
            self.changeDirection1 = 'down'
        
        if event.key == K_RIGHT:
            self.changeDirection2 = 'right'
        if event.key == K_LEFT:
            self.changeDirection2 = 'left'
        if event.key == K_UP:
            self.changeDirection2 = 'up'
        if event.key == K_DOWN:
            self.changeDirection2 = 'down'
        
        if event.key == K_ESCAPE:
            pygame.event.post(pygame.event.Event(QUIT))
        
        if event.key == ord('.') or event.key == ord('>'):
            self.gameSpeed = min(60, self.gameSpeed + 2)
        
        if event.key == ord(',') or event.key == ord('<'):
            self.gameSpeed = max(10, self.gameSpeed - 2)
        
        if event.key == K_SPACE:
            self.pause_game()
    
    def update_game(self):
        if self.changeDirection1 == 'right' and not self.direction1 == 'left':
            self.direction1 = self.changeDirection1
        if self.changeDirection1 == 'left' and not self.direction1 == 'right':
            self.direction1 = self.changeDirection1
        if self.changeDirection1 == 'up' and not self.direction1 == 'down':
            self.direction1 = self.changeDirection1
        if self.changeDirection1 == 'down' and not self.direction1 == 'up':
            self.direction1 = self.changeDirection1
        
        if self.changeDirection2 == 'right' and not self.direction2 == 'left':
            self.direction2 = self.changeDirection2
        if self.changeDirection2 == 'left' and not self.direction2 == 'right':
            self.direction2 = self.changeDirection2
        if self.changeDirection2 == 'up' and not self.direction2 == 'down':
            self.direction2 = self.changeDirection2
        if self.changeDirection2 == 'down' and not self.direction2 == 'up':
            self.direction2 = self.changeDirection2
        
        if self.direction1 == 'right':
            self.snakePosition1[0] += 20
        if self.direction1 == 'left':
            self.snakePosition1[0] -= 20
        if self.direction1 == 'up':
            self.snakePosition1[1] -= 20
        if self.direction1 == 'down':
            self.snakePosition1[1] += 20
        
        if self.direction2 == 'right':
            self.snakePosition2[0] += 20
        if self.direction2 == 'left':
            self.snakePosition2[0] -= 20
        if self.direction2 == 'up':
            self.snakePosition2[1] -= 20
        if self.direction2 == 'down':
            self.snakePosition2[1] += 20
        
        self.snakeSegments1.insert(0, list(self.snakePosition1))
        if self.snakePosition1 in self.candyPosition:
            self.candySpawned = max(0, self.candySpawned - 1)
            self.score1 += 5
            self.scoreColor1 = GOLD
            self.candyPosition.remove(self.snakePosition1)
        else:
            self.snakeSegments1.pop()
            self.scoreColor1 = GREY
        
        self.snakeSegments2.insert(0, list(self.snakePosition2))
        if self.snakePosition2 in self.candyPosition:
            self.candySpawned = max(0, self.candySpawned - 1)
            self.score2 += 5
            self.scoreColor2 = GOLD
            self.candyPosition.remove(self.snakePosition2)
        else:
            self.snakeSegments2.pop()
            self.scoreColor2 = GREY
        
        self.check_collisions()
        
    def render_game(self):
        self.playSurface.fill(BLACK)
        
        if self.candySpawned == 0:
            new_x = random.sample(WIDTH_GRID, self.candyNum)
            new_y = random.sample(HEIGHT_GRID, self.candyNum)
            self.candyPosition = [list(z) for z in zip(new_x, new_y)]
            self.candySpawned = self.candyNum
        
        for cp in self.candyPosition:
            pygame.draw.rect(self.playSurface, RED, Rect(cp[0], cp[1], 20, 20))
        
        for position in self.snakeSegments1[1:]:
            pygame.draw.rect(self.playSurface, self.snakeColor1, Rect(position[0], position[1], 20, 20))
        pygame.draw.rect(self.playSurface, GREY, Rect(self.snakePosition1[0], self.snakePosition1[1], 20, 20))
        
        for position in self.snakeSegments2[1:]:
            pygame.draw.rect(self.playSurface, self.snakeColor2, Rect(position[0], position[1], 20, 20))
        pygame.draw.rect(self.playSurface, GREY, Rect(self.snakePosition2[0], self.snakePosition2[1], 20, 20))
        
        passTime = (pygame.time.get_ticks() - self.gameStartTime) / 1000 - self.pauseTotalTime
        if round(self.gameRestrictTime - passTime) <= 10:
            self.timeColor = RED
            self.candyNum = 10
        
        timeFont = pygame.font.Font(FONT_PATH, 32)
        timeSurf = timeFont.render(str(round(self.gameRestrictTime - passTime)) + 's', True, self.timeColor)
        timeRect = timeSurf.get_rect()
        timeRect.midtop = (WIDTH / 2, 0)
        self.playSurface.blit(timeSurf, timeRect)
        showText(self.playSurface, "1P  " + str(self.score1).zfill(3), FontSize=24, FontColor=self.scoreColor1, midtop=(160, 0))
        showText(self.playSurface, "2P  " + str(self.score2).zfill(3), FontSize=24, FontColor=self.scoreColor2, midtop=(480, 0))
        showText(self.playSurface, "Speed : " + str(self.gameSpeed), FontSize=20, FontColor=self.scoreColor1, midtop=(340, 440))
        showText(self.playSurface, "× " * self.foul1, FontSize=24, FontColor=RED, midtop=(40, 0))
        showText(self.playSurface, "× " * self.foul2, FontSize=24, FontColor=RED, midtop=(600, 0))
        
        pygame.display.flip()

    def check_collisions(self):
        if self.snakePosition1[0] >= WIDTH:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition1[0] = 0
        if self.snakePosition1[0] < 0:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition1[0] = WIDTH - 20
        if self.snakePosition1[1] >= HEIGHT:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition1[1] = 0
        if self.snakePosition1[1] < 0:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition1[1] = HEIGHT - 20
        
        for snakeBody in self.snakeSegments1[1:]:
            if self.snakePosition1 == snakeBody:
                self.foul1 += 1
                if self.foul1 >= 3:
                    self.running = gameOver(self.playSurface, self.score1, self.score2, self.foul1, self.foul2)
                self.snakeColor1 = pygame.Color(max(0, 255 - self.foul1 * 40), max(0, 255 - self.foul1 * 40), 0)
                break
        
        if self.snakePosition2[0] >= WIDTH:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition2[0] = 0
        if self.snakePosition2[0] < 0:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition2[0] = WIDTH - 20
        if self.snakePosition2[1] >= HEIGHT:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition2[1] = 0
        if self.snakePosition2[1] < 0:
            if self.border:
                self.running = gameOver(self.playSurface, self.score1, self.score2)
            else:
                self.snakePosition2[1] = HEIGHT - 20
        
        for snakeBody in self.snakeSegments2[1:]:
            if self.snakePosition2 == snakeBody:
                self.foul2 += 1
                if self.foul2 >= 3:
                    self.running = gameOver(self.playSurface, self.score1, self.score2, self.foul1, self.foul2)
                self.snakeColor2 = pygame.Color(max(0, 135 - self.foul2 * 40), max(0, 206 - self.foul2 * 40), max(0, 235 - self.foul2 * 40))
                break
        
        for snakeBody in self.snakeSegments2[1:]:
            if self.snakePosition1 == snakeBody:
                self.score1 += 5
                self.score2 = max(0, self.score2 - 5)
                self.snakeSegments2.pop()
                self.snakeSegments1.insert(0, list(self.snakePosition1))
                self.snakeColor2 = RED
                self.scoreColor2 = RED
                break
            else:
                self.snakeColor2 = pygame.Color(max(0, 135 - self.foul2 * 40), max(0, 206 - self.foul2 * 40), max(0, 235 - self.foul2 * 40))
                self.scoreColor2 = GREY
        
        for snakeBody in self.snakeSegments1[1:]:
            if self.snakePosition2 == snakeBody:
                self.score2 += 5
                self.score1 = max(0, self.score1 - 5)
                self.snakeSegments1.pop()
                self.snakeSegments2.insert(0, list(self.snakePosition2))
                self.snakeColor1 = RED
                self.scoreColor1 = RED
                break
            else:
                self.snakeColor1 = pygame.Color(max(0, 255 - self.foul1 * 40), max(0, 255 - self.foul1 * 40), 0)
                self.scoreColor1 = GREY

    def pause_game(self):
        pauseStartTime = pygame.time.get_ticks()
        pauseFont = pygame.font.Font(FONT_PATH, 48)
        pauseSurf = pauseFont.render('Pause!', True, RED)
        pauseRect = pauseSurf.get_rect()
        pauseRect.midtop = (320, 125)
        self.playSurface.blit(pauseSurf, pauseRect)
        
        scoreFont = pygame.font.Font(FONT_PATH, 48)
        scoreSurf = scoreFont.render('1P : ' + str(self.score1) + '  v.s  ' + '2P : ' + str(self.score2), True, GREY)
        scoreRect = scoreSurf.get_rect()
        scoreRect.midtop = (320, 225)
        self.playSurface.blit(scoreSurf, scoreRect)
        
        contFont = pygame.font.Font(FONT_PATH, 28)
        contSurf = contFont.render('[Space]:continue    [Esc]:quit', True, BLUE)
        contRect = contSurf.get_rect()
        contRect.midtop = (320, 375)
        self.playSurface.blit(contSurf, contRect)
        
        pygame.display.flip()
        
        pauseCheck = True
        pygame.event.wait()
        while pauseCheck:
            for pauseEvent in pygame.event.get():
                if pauseEvent.type == KEYDOWN:
                    if pauseEvent.key == K_SPACE:
                        pauseCheck = False
                        break
                    elif pauseEvent.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
        
        pauseEndTime = pygame.time.get_ticks()
        self.pauseTotalTime += (pauseEndTime - pauseStartTime) / 1000
