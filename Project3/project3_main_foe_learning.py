import gym
import grid_soccer_NCR
from project3_Q_learning_algorithms import foe_q_learning_agent
import matplotlib.pyplot as plt
import numpy as np
import random as rand

env = gym.make('Grid-Soccer-NCR-v1').unwrapped
N_episodes=100000

player0=foe_q_learning_agent(verbose=False)



for i_episode in range(N_episodes):
    observation = env.reset()
    state = int(str(observation[0])+str(observation[1])+str(observation[2]))
    player0.s=state
    player0.a=2
    action_player_0=2
    action_player_1=4

    for t in range(10000):
        env.render()
        #NC: convert observation to a state ID


        action=(action_player_0,action_player_1)
        # print('action',action)
        # print('observation',observation)

        new_observation, reward, done = env.step(action)

        # print('new observation', new_observation)
        #
        # print('reward', reward)

        #NC: convert observation to a state ID
        new_state = int(str(new_observation[0])+str(new_observation[1])+str(new_observation[2]))

        action_player_0=player0.learn(new_state,action_player_1,reward[0])
        action_player_1=rand.randint(0, 4) #only 0 is leaning, 1 is used to collect data as the game is symmetric, for computational speed up

        #total_reward+=reward

            #reward=max(min(reward,1),-1)
            #agent.remember([observation,action,new_observation,reward,done])
            #agent.train(target,agent_optimizer)
            # if np.random.rand()<=epsilon:
            #     new_action=np.random.randint(0,env.action_space.n)
            # else:
            #
            #     new_action = np.argmax(Q_table[new_observation,:])
            #
            # Q_table[observation,action]+=alpha*(reward+gamma*np.max(Q_table[new_observation,:])-Q_table[observation,action])
            #
        #observation=new_observation
            # action=new_action

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    if i_episode%1000==0:
        print('episode',i_episode)
            #print('player0 Q table', player0.Q_table)



#for_plot=np.array(player0.error_during_training)==0
data_plot=np.array(player0.error_during_training)
plt.figure(figsize=(15,8))
plt.plot(data_plot)
plt.title('Foe Q-learner')
plt.ylabel('Q-value difference')
plt.ylim(0, 0.5)
plt.xlabel('Simulation iteration')
plt.savefig('replication_figure3_foe_q_learning.png', format = 'png')
    #
