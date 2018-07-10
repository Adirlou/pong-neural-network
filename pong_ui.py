import pygame, sys
from pygame.locals import *
from pong_state import *

#Init PyGame
pygame.init()
fps = pygame.time.Clock()


#Set the necessary variables
WHITE = (255,255,255)
RED = (255,0,0)
BLACK = (0,0,0)

WIDTH = 1200
HEIGHT = 800
BALL_RADIUS = 10
PADDLE_WIDTH = int(paddle_width*WIDTH)
PADDLE_HEIGHT = int(paddle_height*HEIGHT)

#Canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Pong Game')

#Draw on given canvas given a certain state
def draw(canvas, state, score):
    #convert paddle and ball position, so that it fits the dimension of the screen of the game
    ballPosition = (int(state.ball_x*WIDTH), int(state.ball_y*HEIGHT))
    paddlePosition = (WIDTH-10, int(state.paddle_y*HEIGHT))

    canvas.fill(WHITE)

    #Draw the paddle and the ball
    pygame.draw.circle(canvas, RED, ballPosition, BALL_RADIUS, 0)
    pygame.draw.line(canvas, BLACK, paddlePosition, (paddlePosition[0], paddlePosition[1] + PADDLE_HEIGHT), 10)

    #display the score
    myFont = pygame.font.SysFont("Comic Sans MS", 40)
    label2 = myFont.render("Score: "+str(score), 1, BLACK)
    canvas.blit(label2, (600, 40))
