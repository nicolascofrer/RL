import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_agent(nn.Module):
    def __init__(self,observation,action):
        #NC: inherit from nn
        super().__init__()
        #NC: experience replay memory
        self.memory_limit=1000000
        self.n_actions=action
        self.observation_memory=np.zeros((self.memory_limit, observation))
        self.action_memory=np.zeros((self.memory_limit, 1))
        self.new_observation_memory=np.zeros((self.memory_limit, observation))
        self.reward_memory=np.zeros((self.memory_limit, 1))
        self.done_memory=np.zeros((self.memory_limit, 1))
        self.replay_memory_limit=self.memory_limit

        self.gamma=0.99
        self.eps=0.99
        #self.eps_decay_rate=1
        self.eps_decay_rate=0.99
        self.memory_index=0
        self.n_iter=0
        self.sample_size=50
        self.update_target=10


        #NC: network
        self.pn1 = nn.Linear(observation, 50)
        #self.pn2 = nn.Linear(50, 50)
        self.pn3 = nn.Linear(50, action)




    def forward(self, observation):
        pn = F.relu(self.pn1(observation))
        #pn = F.relu(self.pn2(pn))



        return self.pn3(pn)

    def remember(self,event):
        #NC: if memory is full, remove the oldest record and add the new one at the end of the list. Order does not matter, later this memory will be sampled uniformly
        #NC: event is (state, next_state, reward, done)
        #print('storing event')
        self.observation_memory[self.memory_index]=event[0]
        self.action_memory[self.memory_index]=event[1]
        self.new_observation_memory[self.memory_index]=event[2]
        self.reward_memory[self.memory_index]=event[3]
        self.done_memory[self.memory_index]=event[4]

        self.memory_index = (self.memory_index + 1) % self.replay_memory_limit

        self.n_iter+=1

    def sample_from_memory(self):
        idx = np.random.randint(0, self.replay_memory_limit, size=self.sample_size)

        return (torch.Tensor(self.observation_memory[idx]),
                torch.LongTensor(self.action_memory[idx]),
                torch.Tensor(self.new_observation_memory[idx]),
                torch.Tensor(self.reward_memory[idx]),
                torch.Tensor(self.done_memory[idx]))



    def take_action(self, observation):
        if random.random()<self.eps:
            #NC: random action
            #print('DQN agent takes random action')
            return random.randint(0,self.n_actions-1)

        else: #NC: return optimal according to policy network
            #print('DQN agent takes greedy action')

            with torch.no_grad():
                #NC: reshape(1, -1) to convert in a np 1 row array
                return self.forward(torch.Tensor(observation.reshape(1, -1))).max(1)[1].detach().numpy().flatten()[0]

        self.eps*=self.eps_decay_rate

    def train(self, target_network, agent_optimizer):
        #NC: wait for memory to be full and then learn
        if self.n_iter>self.replay_memory_limit:
            #NC: for the sample from the memory, we need to estimate Q(s,a) according to the policy network
            observation, action, new_observation, reward, done=self.sample_from_memory()
            #with torch.no_grad():

            Q_policy=self.forward(observation).gather(1, action)

                #NC: if done, the continuation value is 0
            with torch.no_grad():
                Q_target=reward + self.gamma*(1-done)*target_network(new_observation).max(1)[0].reshape(-1, 1)




                #print('Q_target',Q_target)
                #print('new_observation',new_observation)

                #print('target_network(new_observation).max(1)[0]',target_network(new_observation).max(1)[0])
                #print('shape',target_network(new_observation).max(1)[0].shape)
                #Q_target+=self.gamma*(1-done)*target_network(new_observation).max(1)[0]



            #NC: after some iterations, update the target copying the actual policy
            if self.n_iter % self.update_target == 0:
                #print('iter ',self.n_iter)
                #print('updating target')
                #NC: we copy the parameters of the policy to the target
                target_network.load_state_dict(self.state_dict())

            loss = F.mse_loss(Q_policy, Q_target)
            #print('Q_policy',Q_policy)
            #print('Q_target',Q_target)

            agent_optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            agent_optimizer.step()

            self.eps*=self.eps_decay_rate






#agent=DQN_agent(state_dimension,action_dimesion,memory_limit)
#optimizer = torch.optim.Adam(agent.parameters(),lr=3e-4)
