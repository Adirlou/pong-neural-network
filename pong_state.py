from enum import Enum
import math
import random

paddle_height = 0.2
paddle_width = 0.00

class Action(Enum):
    nothing = 0
    up = 1
    down = 2

"""
Represents a state of the environment
"""
class State:
    def __init__(self, ball_x, ball_y, v_x, v_y, paddle_y, reward = 0):
        self.reward = reward

        #Set the different attributes
        if ball_x < 0:
            self.ball_x = 0
        else:
            self.ball_x = ball_x

        if ball_y < 0:
            self.ball_y = 0
        elif ball_y > 1:
            self.ball_y = 1
        else:
            self.ball_y = ball_y

        if abs(v_x) < 0.03:
            if v_x != 0:
                self.velocity_x = 0.03 * (v_x / abs(v_x));
            else:
                self.velocity_x = 0
        else:
            self.velocity_x = v_x
        self.velocity_y = v_y

        if paddle_y < 0:
            self.paddle_y = 0
        elif paddle_y > 1 - paddle_height:
            self.paddle_y = 1 - paddle_height
        else:
            self.paddle_y = paddle_y

    """
    Returns a list containing the attributes of a State
    """
    def returnListofAttributes(self):
        return [self.ball_x, self.ball_y, self.velocity_x, self.velocity_y, self.paddle_y]

    """
    Returns a random value uniformly between -v and v
    """
    @staticmethod
    def uniform(v):
        return (random.random() - 0.5) * v

    """
    Returns the initial state
    """
    @staticmethod
    def initialState():
        return State(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2)

    """
    Retuns a game-over state
    """
    @staticmethod
    def lost():
        s = State(0,0,0,0,0)
        s.ball_x = -1
        return s

    """
    Returns whether the game is over
    """
    def gameOver(self):
        return self.ball_x > 1 or self.ball_x == -1

    """
    Returns the next state given the provided action
    """
    def nextState(self, action):
        next_bx = self.ball_x + self.velocity_x
        next_by = self.ball_y + self.velocity_y
        next_vx = self.velocity_x
        next_vy = self.velocity_y
        next_reward = 0
        next_py = self.paddle_y
        if action == Action.up:
            next_py -= 0.04
        elif action == Action.down:
            next_py += 0.04

        #Take care of when the ball hits one of the walls
        if next_by < 0 :
            next_by *= -1
            next_vy *= -1
        elif next_by > 1:
            next_by = 2 - next_by
            next_vy *= -1
        if next_bx < 0:
            next_bx *= -1
            next_vx *= -1
        elif next_bx > 1:#in this case we may be winning

            if next_by > next_py and next_by < next_py + paddle_height:#we win
                next_reward = 1
                next_bx = 2 - next_bx
                next_vx = - next_vx + State.uniform(0.015)
                next_vy = next_vy + State.uniform(0.03)
            else:#we lost
                next_reward = -1
                s = State.lost();
                s.reward = -1
                return s

        return State(next_bx, next_by, next_vx, next_vy, next_py, next_reward)

    def __repr__(self):
        return "State(ball : ("+str(self.ball_x)+", "+str(self.ball_y)+"), velocity : ("+str(self.velocity_x)+", "+str(self.velocity_y)+"), paddle : (1, "+str(self.paddle_y)+")"

    def __hash__(self):
        return hash(self.ball_x) + hash(self.ball_y) + hash(self.velocity_x) + hash(self.velocity_y) + hash(self.paddle_y)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.ball_x == other.ball_x and self.ball_y == other.ball_y and self.velocity_x == other.velocity_x and self.velocity_y == other.velocity_y and self.paddle_y == other.paddle_y
        else:
            return False
    def __ne__(self, other):
        return not self.__eq__(other)
