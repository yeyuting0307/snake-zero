import os
import sys
import time
import random
import pygame
from pygame.locals import *
import numpy as np
from constants import *
from vcr import EpisodeRecorder

class Game:
    def __init__(self, playSurface, p1_auto=False, p2_auto=False):
        self.ps = playSurface
        self.fpsClock = pygame.time.Clock()
        self.recorder = EpisodeRecorder()
        self.reset()
        self.p1_auto = p1_auto
        self.p2_auto = p2_auto
        
    def reset(self):
        self.reset_state()
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
        self.gameWinScore = 100
        self.gameRestrictTime = 42
        self.gameSpeed = 16
        self.maxSpeed = 60
        self.minSpeed = 10
        self.pauseTotalTime = 0
        self.running = True
        self.breakRule = False

    def reset_state(self):
        self.if_record = False
        self._state = np.zeros((HEIGHT, WIDTH, 3))
        self._action1 = None
        self._action2 = None
        self._reward1 = 0
        self._reward2 = 0
        self._next_state = np.zeros((HEIGHT, WIDTH, 3))
        self._terminated = False
        self._truncated = False
        self._info = None

    def run(self):
        holdStart = True
        while holdStart:
            holdStart = self.gameStart()
        
        self.ps.fill(BLACK)
        ruleStart = True
        while ruleStart:
            ruleStart = self.gameRule()
        
        restart = True
        while restart:
            self.recorder.reset()
            self.reset()
            self.gameStartTime = pygame.time.get_ticks()
            
            while self.running and not self.breakRule:
                self.auto_play(self.p1_auto, self.p2_auto)
                self._state = pygame.surfarray.array3d(self.ps)
                self._action = None
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == KEYDOWN:
                        self.if_record = True
                        self.handle_keydown(event)
                        
                self.update_game()
                self.check_collisions()
                self._info = (self.score1, self.score2)
                self._truncated = self.breakRule

                self.render_game()
                self._next_state = pygame.surfarray.array3d(self.ps)
                
                self.check_win()
                self.check_time_up()
                self._terminated = not self.running

                self.check_record()
                self.fpsClock.tick(self.gameSpeed)

    def check_win(self):
        if self.score1 >= self.gameWinScore or self.score2 >= self.gameWinScore:
            self.running = False
            self.gameOver(self.score1, self.score2)
            
    def check_record(self):
        self.recorder.record_buffer(
            self._state, self._action1, self._action2, 
            self._reward1, self._reward2, self._next_state, 
            self._terminated, self._truncated, self._info
        )
        if self.if_record:
            self.recorder.record_state()
            self.reset_state()

    def check_time_up(self):
        passTime = (pygame.time.get_ticks() - self.gameStartTime) / 1000 - self.pauseTotalTime
        if self.gameRestrictTime - passTime <= 0:
            self.running = False
            self.gameOver(self.score1, self.score2)

    def handle_keydown(self, event):
        if event.key == ord('d'):
            self._action1 = self.changeDirection1 = 'right'
        if event.key == ord('a'):
            self._action1 = self.changeDirection1 = 'left'
        if event.key == ord('w'):
            self._action1 = self.changeDirection1 = 'up'
        if event.key == ord('s'):
            self._action1 = self.changeDirection1 = 'down'
        
        if event.key == K_RIGHT:
            self._action2 = self.changeDirection2 = 'right'
        if event.key == K_LEFT:
            self._action2 = self.changeDirection2 = 'left'
        if event.key == K_UP:
            self._action2 = self.changeDirection2 = 'up'
        if event.key == K_DOWN:
            self._action2 = self.changeDirection2 = 'down'
        
        if event.key == K_ESCAPE:
            pygame.event.post(pygame.event.Event(QUIT))
        
        if event.key == ord('.') or event.key == ord('>'):
            self.gameSpeed = min(self.maxSpeed, self.gameSpeed + 2)
        
        if event.key == ord(',') or event.key == ord('<'):
            self.gameSpeed = max(self.minSpeed, self.gameSpeed - 2)
        
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
            self._reward1 = 5
            self.scoreColor1 = GOLD
            self.candyPosition.remove(self.snakePosition1)
        else:
            self.snakeSegments1.pop()
            self.scoreColor1 = GREY
        
        self.snakeSegments2.insert(0, list(self.snakePosition2))
        if self.snakePosition2 in self.candyPosition:
            self.candySpawned = max(0, self.candySpawned - 1)
            self.score2 += 5
            self._reward2 = 5
            self.scoreColor2 = GOLD
            self.candyPosition.remove(self.snakePosition2)
        else:
            self.snakeSegments2.pop()
            self.scoreColor2 = GREY
        
    def render_game(self):
        self.ps.fill(BLACK)
        
        if self.candySpawned == 0:
            new_x = random.sample(WIDTH_GRID, self.candyNum)
            new_y = random.sample(HEIGHT_GRID, self.candyNum)
            self.candyPosition = [list(z) for z in zip(new_x, new_y)]
            self.candySpawned = self.candyNum
        
        for cp_x, cp_y in self.candyPosition:
            pygame.draw.rect(self.ps, RED, Rect(cp_x, cp_y, 20, 20))
        
        for position in self.snakeSegments1[1:]:
            pygame.draw.rect(self.ps, self.snakeColor1, Rect(position[0], position[1], 20, 20))
        pygame.draw.rect(self.ps, GREY, Rect(self.snakePosition1[0], self.snakePosition1[1], 20, 20))
        
        for position in self.snakeSegments2[1:]:
            pygame.draw.rect(self.ps, self.snakeColor2, Rect(position[0], position[1], 20, 20))
        pygame.draw.rect(self.ps, GREY, Rect(self.snakePosition2[0], self.snakePosition2[1], 20, 20))
        
        passTime = (pygame.time.get_ticks() - self.gameStartTime) / 1000 - self.pauseTotalTime
        if round(self.gameRestrictTime - passTime) <= 10:
            self.timeColor = RED
            self.candyNum = 10
        
        timeFont = pygame.font.Font(FONT_PATH, 32)
        timeSurf = timeFont.render(str(round(self.gameRestrictTime - passTime)) + 's', True, self.timeColor)
        timeRect = timeSurf.get_rect()
        timeRect.midtop = (WIDTH / 2, 0)
        self.ps.blit(timeSurf, timeRect)
        self.showText("1P  " + str(self.score1).zfill(3), 24, self.scoreColor1, (160, 0))
        self.showText("2P  " + str(self.score2).zfill(3), 24, self.scoreColor2, (480, 0))
        self.showText("Speed : " + str(self.gameSpeed), 20, self.scoreColor1, (340, 440))
        self.showText("× " * self.foul1, 24, RED, (40, 0))
        self.showText("× " * self.foul2, 24, RED, (600, 0))
        pygame.display.flip()

    def check_collisions(self):
        if self.snakePosition1[0] >= WIDTH:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition1[0] = 0
        if self.snakePosition1[0] < 0:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition1[0] = WIDTH - 20
        if self.snakePosition1[1] >= HEIGHT:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition1[1] = 0
        if self.snakePosition1[1] < 0:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition1[1] = HEIGHT - 20
        
        for snakeBody in self.snakeSegments1[1:]:
            if self.snakePosition1 == snakeBody:
                self.foul1 += 1
                if self.foul1 >= 3:
                    self.breakRule = self.gameOver( self.score1, self.score2, self.foul1, self.foul2)
                self.snakeColor1 = pygame.Color(max(0, 255 - self.foul1 * 40), max(0, 255 - self.foul1 * 40), 0)
                break
        
        if self.snakePosition2[0] >= WIDTH:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition2[0] = 0
        if self.snakePosition2[0] < 0:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition2[0] = WIDTH - 20
        if self.snakePosition2[1] >= HEIGHT:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition2[1] = 0
        if self.snakePosition2[1] < 0:
            if self.border:
                self.running = self.gameOver( self.score1, self.score2)
            else:
                self.snakePosition2[1] = HEIGHT - 20
        
        for snakeBody in self.snakeSegments2[1:]:
            if self.snakePosition2 == snakeBody:
                self.foul2 += 1
                if self.foul2 >= 3:
                    self.breakRule = self.gameOver( self.score1, self.score2, self.foul1, self.foul2)
                self.snakeColor2 = pygame.Color(max(0, 135 - self.foul2 * 40), max(0, 206 - self.foul2 * 40), max(0, 235 - self.foul2 * 40))
                break
        
        for snakeBody in self.snakeSegments2[1:]:
            if self.snakePosition1 == snakeBody:
                self.score1 += 5
                self._reward1 = 5
                self.score2 = max(0, self.score2 - 5)
                if self.score2 > 0:
                    self._reward2 = -5
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
                self._reward2 = 5
                self.score1 = max(0, self.score1 - 5)
                if self.score1 > 0:
                    self._reward1 = -5
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
        self.ps.blit(pauseSurf, pauseRect)
        
        scoreFont = pygame.font.Font(FONT_PATH, 48)
        scoreSurf = scoreFont.render('1P : ' + str(self.score1) + '  v.s  ' + '2P : ' + str(self.score2), True, GREY)
        scoreRect = scoreSurf.get_rect()
        scoreRect.midtop = (320, 225)
        self.ps.blit(scoreSurf, scoreRect)
        
        contFont = pygame.font.Font(FONT_PATH, 28)
        contSurf = contFont.render('[Space]:continue    [Esc]:quit', True, BLUE)
        contRect = contSurf.get_rect()
        contRect.midtop = (320, 375)
        self.ps.blit(contSurf, contRect)
        
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

    def showText(self, text, FontSize=72, FontColor=BLUE, midtop=(320, 125), Font=FONT_PATH):
        font = pygame.font.Font(Font, FontSize)
        surf = font.render(text, True, FontColor)
        rect = surf.get_rect()
        rect.midtop = midtop
        self.ps.blit(surf, rect)

    def gameStart(self):
        self.showText('Snake Game', FontSize=72, FontColor=TIFFANY, midtop=(320, 125))
        self.showText('Powered by Mike', FontSize=24, FontColor=ORANGE, midtop=(320, 225))
        self.showText('[Space]:start    [Esc]:quit.', FontSize=28, FontColor=BLUE, midtop=(320, 325))

        pygame.display.flip()
        
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.event.post(pygame.event.Event(QUIT))
                    if event.key == K_SPACE:
                        return False

    def gameRule(self):
        self.showText('【1P】', FontSize=36, FontColor=YELLOW, midtop=(200, 25))
        self.showText('up : [w]', FontSize=28, FontColor=YELLOW, midtop=(200, 75))
        self.showText('left : [a]', FontSize=28, FontColor=YELLOW, midtop=(200, 125))
        self.showText('down : [s]', FontSize=28, FontColor=YELLOW, midtop=(200, 175))
        self.showText('right : [d]', FontSize=28, FontColor=YELLOW, midtop=(200, 225))
        
        self.showText('【2P】', FontSize=36, FontColor=SKYBLUE, midtop=(400, 25))
        self.showText('up : [↑]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 75))
        self.showText('left : [←]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 125))
        self.showText('down : [↓]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 175))
        self.showText('right : [→]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 225))
        
        self.showText("※ Eat red candy or opponent's body to get points.", 
                FontSize=24, FontColor=RED, midtop=(320, 280))
        self.showText("※ Don't eat yourself up to three times or you lose.", 
                FontSize=24, FontColor=RED, midtop=(320, 320))
        
        self.showText('[>] : speed up    [<] : speed down', 
                FontSize=24, FontColor=ORANGE, midtop=(320, 380))
        
        self.showText('[Space] : start/pause             [Esc] : quit', 
                FontSize=24, FontColor=SKYBLUE, midtop=(320, 420))
        
        pygame.display.flip()
        
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.event.post(pygame.event.Event(QUIT))
                    if event.key == K_SPACE:
                        return False

    def gameOver(self, score1, score2, foul1=0, foul2=0):
        if foul1 >= 3:
            endSpeech = '2P Wins!'
            otherSpeech = '1P Break Rule'
        elif foul2 >= 3:
            endSpeech = '1P Wins!'
            otherSpeech = '2P Break Rule'
        elif score1 > score2:
            endSpeech = '1P Wins!'
            otherSpeech = None
        elif score1 < score2:
            endSpeech = '2P Wins!'
            otherSpeech = None
        else:
            endSpeech = 'Break Even!'
            otherSpeech = None
        
        self.showText(endSpeech, FontSize=72, FontColor=RED, midtop=(320, 125))
        self.showText(f'1P : {score1}  v.s  2P : {score2}', FontSize=48, FontColor=LIGHTGREY, midtop=(320, 225))
        
        if otherSpeech:
            self.showText(otherSpeech, FontSize=28, FontColor=RED, midtop=(320, 300))
        
        self.showText('[Space]:restart    [Esc]:quit', FontSize=28, FontColor=BLUE, midtop=(320, 375))
        
        pygame.display.flip()
        self._next_state = pygame.surfarray.array3d(self.ps)

        self.if_record = True
        self.check_record()
        self.recorder.save_episode()
        
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.event.post(pygame.event.Event(QUIT))
                    if event.key == K_SPACE:
                        return False
    def auto_play(self, player1 = False, player2 = False):
        for cp in self.candyPosition:
            if player1:
                if self.snakePosition1[0] > cp[0]:
                    key_event1 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=0, unicode='a')
                elif self.snakePosition1[1] > cp[1]:
                    key_event1 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w, mod=0, unicode='w')
                elif self.snakePosition1[0] < cp[0]:
                    key_event1 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0, unicode='d')
                elif self.snakePosition1[1] < cp[1]:
                    key_event1 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0, unicode='s')
                else:
                    key_event1 =pygame.event.Event(pygame.NOEVENT)
                pygame.event.post(key_event1)
            if player2:
                if self.snakePosition2[0] < cp[0]:
                    key_event2 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0, unicode='right')
                elif self.snakePosition2[0] > cp[0]:
                    key_event2 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT, mod=0, unicode='left')
                elif self.snakePosition2[1] < cp[1]:
                    key_event2 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0, unicode='down')
                elif self.snakePosition2[1] > cp[1]:
                    key_event2 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0, unicode='up')
                else:
                    key_event2 =pygame.event.Event(pygame.NOEVENT)
                pygame.event.post(key_event2)