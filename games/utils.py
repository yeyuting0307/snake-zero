import pygame
import sys
from pygame.locals import *
from constants import *

def showText(playSurface, text, FontSize=72, FontColor=BLUE, midtop=(320, 125), Font=FONT_PATH):
    font = pygame.font.Font(Font, FontSize)
    surf = font.render(text, True, FontColor)
    rect = surf.get_rect()
    rect.midtop = midtop
    playSurface.blit(surf, rect)

def gameStart(playSurface):
    showText(playSurface, 'Snake Game', FontSize=72, FontColor=TIFFANY, midtop=(320, 125))
    showText(playSurface, 'Powered by Mike', FontSize=24, FontColor=ORANGE, midtop=(320, 225))
    showText(playSurface, '[Space]:start    [Esc]:quit.', FontSize=28, FontColor=BLUE, midtop=(320, 325))

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

def gameRule(playSurface):
    showText(playSurface, '【1P】', FontSize=36, FontColor=YELLOW, midtop=(200, 25))
    showText(playSurface, 'up : [w]', FontSize=28, FontColor=YELLOW, midtop=(200, 75))
    showText(playSurface, 'left : [a]', FontSize=28, FontColor=YELLOW, midtop=(200, 125))
    showText(playSurface, 'down : [s]', FontSize=28, FontColor=YELLOW, midtop=(200, 175))
    showText(playSurface, 'right : [d]', FontSize=28, FontColor=YELLOW, midtop=(200, 225))
    
    showText(playSurface, '【2P】', FontSize=36, FontColor=SKYBLUE, midtop=(400, 25))
    showText(playSurface, 'up : [↑]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 75))
    showText(playSurface, 'left : [←]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 125))
    showText(playSurface, 'down : [↓]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 175))
    showText(playSurface, 'right : [→]', FontSize=28, FontColor=SKYBLUE, midtop=(400, 225))
    
    showText(playSurface, "※ Eat red candy or opponent's body to get points.", 
             FontSize=24, FontColor=RED, midtop=(320, 280))
    showText(playSurface, "※ Don't eat yourself up to three times or you lose.", 
             FontSize=24, FontColor=RED, midtop=(320, 320))
    
    showText(playSurface, '[>] : speed up    [<] : speed down', 
             FontSize=24, FontColor=ORANGE, midtop=(320, 380))
    
    showText(playSurface, '[Space] : start/pause             [Esc] : quit', 
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

def gameOver(playSurface, score1, score2, foul1=0, foul2=0):
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
    
    showText(playSurface, endSpeech, FontSize=72, FontColor=RED, midtop=(320, 125))
    showText(playSurface, f'1P : {score1}  v.s  2P : {score2}', FontSize=48, FontColor=LIGHTGREY, midtop=(320, 225))
    
    if otherSpeech:
        showText(playSurface, otherSpeech, FontSize=28, FontColor=RED, midtop=(320, 300))
    
    showText(playSurface, '[Space]:restart    [Esc]:quit', FontSize=28, FontColor=BLUE, midtop=(320, 375))
    
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