import numpy as np
import random as rand
from cvxopt import matrix, solvers

class sarsa_agent():

    def __init__(self,verbose=True):
        self.verbose=verbose

        if self.verbose: print("starting sarsa")
        self.alpha=1
        self.gamma=0.9
        self.epsilon=1
        self.decay_rate_epsilon=np.e**(np.log(0.001)/1e6)
        self.decay_rate_alpha=np.e**(np.log(0.001)/1e6)


        self.Q_table=np.zeros((773,5))
        self.s = 2
        self.a = 4
        self.error_during_training=[]


    def get_action(self, s):

        #self.s = s

        #NC: random action, e-greedy policy
        if rand.random()<self.epsilon:
            action = rand.randint(0, 4)
        else:
            action= np.argmax(self.Q_table[s,:])


        if self.verbose: print("sarsa taking action",action)


        return action

    def learn(self,new_s,r):
        #NC: Updating Q_table
        r=(1-self.gamma)*r
        old_Q_table_value_for_graph=self.Q_table[211,2]
        if self.verbose: print("sarsa learning from new transition")
        if self.verbose: print("learning with reward",r)

        new_action=self.get_action(new_s)

        new_Q = r + self.gamma*self.Q_table[new_s,new_action]


        self.Q_table[self.s,self.a]+=self.alpha*(new_Q-self.Q_table[self.s,self.a])


        updated_Q_table_value_for_graph=self.Q_table[211,2]

        self.error_during_training.append(abs(old_Q_table_value_for_graph-updated_Q_table_value_for_graph))
        #update state and action for next state

        self.s=new_s
        self.a=new_action
        self.epsilon=self.epsilon*self.decay_rate_epsilon
        self.alpha=self.alpha*self.decay_rate_alpha

        return new_action




class foe_q_learning_agent():

    def __init__(self,verbose=True):
        self.verbose=verbose

        if self.verbose: print("starting foe Q learning agent")
        self.alpha=1
        self.gamma=0.9
        self.epsilon=1.0
        self.decay_rate_epsilon=1
        self.decay_rate_alpha=np.e**(np.log(0.001)/1e6)


        # self.Q_table=np.random.rand(773,5,5)*10
        self.Q_table=np.zeros((773,5,5))

        self.V_table=np.zeros(773)
        # self.V_table=np.random.rand(773)*10

        self.s = 211
        self.a = 2

        self.error_during_training=[]
        self.x=[1/5,1/5,1/5,1/5,1/5]

    def get_action(self, s):

        #self.s = s



        #NC: random action, at the end it is only used to gather data from the environment
        action = rand.randint(0, 4)


        if self.verbose: print("foe learner taking random action",action)


        return action


    def solve_minimax(self,game):
        A=matrix([1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
             -game[0,0],-game[1,0],-game[2,0],-game[3,0],-game[4,0],1.0,-1.0,-1.0,0.0,0.0,0.0,0.0,
             -game[0,1],-game[1,1],-game[2,1],-game[3,1],-game[4,1],1.0,-1.0,0.0,-1.0,0.0,0.0,0.0,
             -game[0,2],-game[1,2],-game[2,2],-game[3,2],-game[4,2],1.0,-1.0,0.0,0.0,-1.0,0.0,0.0,
             -game[0,3],-game[1,3],-game[2,3],-game[3,3],-game[4,3],1.0,-1.0,0.0,0.0,0.0,-1.0,0.0,
             -game[0,4],-game[1,4],-game[2,4],-game[3,4],-game[4,4],1.0,-1.0,0.0,0.0,0.0,0.0,-1.0


             ],(12,6))
        print('A\n',A)
        b=matrix([ 0.0,0.0,0.0,0.0,0.0,1.0,-1.0 ,0.0,0.0,0.0,0.0,0.0])
        print('b\n',b)
        c = matrix([ -1.0,0.0,0.0,0.0,0.0,0.0])
        print('c\n',c)

        sol=solvers.lp(c,A,b,solver="glpk")
        print('sol\n', sol['x'])
        return sol['x']

    def learn(self,new_s,opponent_action,r):
        #NC: Updating Q_table
        # print('updating foe q beginning')

        # print('self.a',self.a)
        # print('self.s',self.s)

        # print('new_s',new_s)
        # print('opponent_action',opponent_action)

        r=(1-self.gamma)*r
        old_Q_table_value_for_graph=self.Q_table[211,2,4]
        if self.verbose: print("foe q learning from new transition")
        if self.verbose: print("learning with reward",r)


        new_action=self.get_action(new_s)

        new_Q = r + self.gamma*self.V_table[new_s]


        self.Q_table[self.s,self.a,opponent_action]+=self.alpha*(new_Q-self.Q_table[self.s,self.a,opponent_action])

        #print('self.Q_table[self.s,:,:]',self.Q_table[self.s,:,:].shape)

        self.x=self.solve_minimax(self.Q_table[self.s,:,:])
        try:
            self.V_table[self.s]=self.x[0]
        except:
            print('NO SOLUTION FOUND')

        updated_Q_table_value_for_graph=self.Q_table[211,2,4]
        # print('learning in foe q')
        # print('state',self.s)
        # print('action',self.a)
        # print('opponent_action',opponent_action)


        # if self.s==211 and self.a==2 and opponent_action==4:
        #     print("HERE")
        #     print('state',self.s)
        #     print('updated Q',self.Q_table[211,2,4])

        self.error_during_training.append(abs(old_Q_table_value_for_graph-updated_Q_table_value_for_graph))
        #update state and action for next state

        self.s=new_s
        self.a=new_action
        self.epsilon=self.epsilon*self.decay_rate_epsilon
        self.alpha=self.alpha*self.decay_rate_alpha

        return new_action

class friend_q_learner():

    def __init__(self,verbose=True):
        self.verbose=verbose

        if self.verbose: print("starting friend q")
        self.alpha=1
        self.gamma=0.9
        self.epsilon=1
        self.decay_rate_epsilon=np.e**(np.log(0.001)/1e6)
        self.decay_rate_alpha=np.e**(np.log(0.001)/1e6)




        #NC: for friend q i use a map from actions pair to unique action ID
        self.Q_table=np.zeros((773,45))
        self.s = None
        self.a = None
        self.error_during_training=[]


    def get_action(self, s):

        #self.s = s

        #NC: random action, e-greedy policy
        if rand.random()<self.epsilon:
            action = rand.randint(0, 44)
        else:
            action= np.argmax(self.Q_table[s,:])


        if self.verbose: print("friend q taking action",action)


        return action

    def learn(self,new_s,r,actions):
        #NC: Updating Q_table
        r=(1-self.gamma)*r

        old_Q_table_value_for_graph=self.Q_table[211,24]
        if self.verbose: print("friend q learning from new transition")
        if self.verbose: print("learning with reward",r)

        new_action=self.get_action(new_s)

        new_Q = r + self.gamma*np.max(self.Q_table[new_s,:])


        self.Q_table[self.s,actions]+=self.alpha*(new_Q-self.Q_table[self.s,actions])





        updated_Q_table_value_for_graph=self.Q_table[211,24]

        self.error_during_training.append(abs(old_Q_table_value_for_graph-updated_Q_table_value_for_graph))
        #update state and action for next state

        self.s=new_s
        self.a=new_action
        self.epsilon=self.epsilon*self.decay_rate_epsilon
        self.alpha=self.alpha*self.decay_rate_alpha

        return new_action
