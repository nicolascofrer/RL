#import os, subprocess, time, signal
import gym
from gym import spaces
import numpy as np
#from gym import error, spaces
#from gym import utils
#from gym.utils import seeding

class grid_soccer_NCR(gym.Env):
    #NC: player A is player 0, player B is player 1
    metadata = {'render.modes': ['human']}

    def __init__(self):


        self.lastaction = None
        #NCR: action for P0 and action for P1
        self.action_space = spaces.Tuple((spaces.Discrete(5), spaces.Discrete(5)))
        #NCR: position of player 0, player 1 and who has the ball
        self.observation_space = spaces.Tuple((spaces.Discrete(8), spaces.Discrete(8),spaces.Discrete(2)))
        #NCR: initial state is random
        pos_0=spaces.Discrete(8).sample()
        pos_1=spaces.Discrete(8).sample()
        while pos_0==pos_1 or pos_0==0 or pos_0==4 or pos_0==3 or pos_0==7 or pos_1==0 or pos_1==4 or pos_1==3 or pos_1==7:
            pos_1=spaces.Discrete(8).sample()
            pos_0=spaces.Discrete(8).sample()

        self.s = (pos_0,pos_1,spaces.Discrete(2).sample())
        self.done=False


    def reset(self):
        #NCR: initial state is random
        pos_0=spaces.Discrete(8).sample()
        pos_1=spaces.Discrete(8).sample()
        while pos_0==pos_1 or pos_0==0 or pos_0==4 or pos_0==3 or pos_0==7 or pos_1==0 or pos_1==4 or pos_1==3 or pos_1==7:
            pos_1=spaces.Discrete(8).sample()
            pos_0=spaces.Discrete(8).sample()

        self.s = (pos_0,pos_1,spaces.Discrete(2).sample())
        self.lastaction = None
        self.done=False

        #force initial state as example state in paper
        self.s = (2,1,1)

        return self.s

    def action_mapper(self,a):
        if a==0:
            return -4
        elif a==1:
            return 1
        elif a==2:
            return 4
        elif a==3:
            return -1
        else:
            return 0

    def step(self, a):
        #NCR: new_observation, reward, done

        s0,s1,ball=self.s
        #NCR: who plays first?
        first_mover=np.random.randint(0,2)


        #print('a[0]',a[0])
        #if first_mover==0:
        new_pos0=s0+self.action_mapper(a[0])
        new_pos1=s1+self.action_mapper(a[1])

        #NC: check collisions
        if new_pos0==new_pos1:
            #NC: only first moves only if valid move
            if first_mover==0 and new_pos0>=0 and new_pos0<=7:
                s0=new_pos0
                #NC: if ball is in second mover, change posesion
                if ball==1: ball=0
            elif new_pos1>=0 and new_pos1<=7:
                s1=new_pos1
                #NC: if ball is in second mover, change posesion
                if ball==0: ball=1

        else: #NC: no collision, if they are valid moves, then move
            if new_pos0>=0 and new_pos0<=7:
                s0=new_pos0
            if new_pos1>=0 and new_pos1<=7:
                s1=new_pos1

        r=(0,0)

        if (self.s[0]==3 or self.s[0]==7) and self.s[2]==0:

            self.done=True
            r=(100,-100)

        if (self.s[0]==0 or self.s[0]==4) and self.s[2]==0:

            self.done=True
            r=(-100,100)

        if (self.s[1]==0 or self.s[1]==4) and self.s[2]==1:

            r=(-100,100)
            self.done=True

        if (self.s[1]==3 or self.s[1]==7) and self.s[2]==1:

            r=(100,-100)
            self.done=True

        self.lastaction = a
        self.s=(s0,s1,ball)
        return (self.s, r, self.done)
    def render(self):
        pass
