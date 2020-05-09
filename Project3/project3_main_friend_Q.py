import gym
import grid_soccer_NCR
from project3_Q_learning_algorithms import friend_q_learner
import matplotlib.pyplot as plt
import numpy as np
import random as rand

env = gym.make('Grid-Soccer-NCR-v1').unwrapped
N_episodes=150000

player0=friend_q_learner(verbose=False)


for i_episode in range(N_episodes):
    observation = env.reset()
    state = int(str(observation[0])+str(observation[1])+str(observation[2]))
    #actions0=player0.get_action(state)
    action_player_0=2
    # print('episode',i_episode)
    # print('observation',observation)
    # print('state',state)
    #

    action_player_1=4

    for t in range(10000):
        env.render()
        #NC: convert observation to a state ID


        action=(action_player_0,action_player_1)

        new_observation, reward, done = env.step(action)





        #NC: convert observation to a state ID
        new_state = int(str(new_observation[0])+str(new_observation[1])+str(new_observation[2]))
        # print('new_state',new_state)
        actions0=player0.learn(new_state,reward[0],int(str(action[0])+str(action[1])))
        # actions1=player1.learn(new_state,reward[1])

        action_player_0=int(str(actions0).zfill(2)[0])
        action_player_1=rand.randint(0, 4)
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

            break

    if i_episode%1000==0:
        print('episode',i_episode)





#for_plot=np.array(player0.error_during_training)==0
data_plot=np.array(player0.error_during_training)
plt.figure(figsize=(15,8))
plt.plot(data_plot)
plt.title('Friend-Q')
plt.ylabel('Q-value difference')
plt.ylim(0, 0.5)
plt.xlabel('Simulation iteration')
plt.savefig('replication_figure3_friend_q.png', format = 'png')
    #
