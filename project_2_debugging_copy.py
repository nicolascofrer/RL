import gym
import numpy as np
import matplotlib.pyplot as plt
import NC_DQN_implementation as DQN_NC
import copy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sarsa_agent(gamma,
alpha,
epsilon,
n_episodes,
seed):

    N_episodes=n_episodes
    random_seed=seed
    #env = gym.make('FrozenLake-v0')
    #input_string='SFFFHFFFFFFFFFFG'
    # map=np.asarray(input_string,dtype='c')
    # map=map.reshape(int(np.sqrt(len(input_string))),int(np.sqrt(len(input_string))))
    # print(map.shape)

    env = gym.make('CartPole-v0').unwrapped

    agent=DQN_NC.DQN_agent(env.observation_space.shape[0],env.action_space.n).to(device)
    agent_optimizer=torch.optim.RMSprop(agent.parameters(), lr=1e-2)

    #agent.load_state_dict(torch.load("agent"))
    #agent_optimizer.load_state_dict(torch.load("agent_optimizer"))


    target=copy.deepcopy(agent)
    episodes_rewards = np.zeros(N_episodes)
    last_100_mean = np.zeros(N_episodes)

    #env.reset()
    #env = gym.make('CartPole-v0')
    # print('env.action_space',env.action_space)
    # print('env.observation_space',env.observation_space)

    # print((env.observation_space.n))
    #alpha=0.25
    #gamma=1.0
    #random_seed=741684
    #epsilon=0.29
    #N_episodes=14697

    # Q_table=np.zeros((env.observation_space.n,env.action_space.n))

    #np.random.seed(random_seed)
    #env.seed(random_seed)
    for i_episode in range(N_episodes):
        observation = env.reset()
        total_reward=0
        # if np.random.rand()<=epsilon:

        #action=np.random.randint(0,env.action_space.n)
        # else:


            # action = np.argmax(Q_table[observation,:])

        for t in range(50000):
            env.render()
            #print(observation)



            # action=np.random.randint(0,env.action_space.n)
            action=agent.take_action(observation)
            #print('DQN takes this action',action)

            #action=action.data.numpy().flatten()
            new_observation, reward, done, info = env.step(action)


            total_reward+=reward

            #reward=max(min(reward,1),-1)
            agent.remember([observation,action,new_observation,reward,done])
            agent.train(target,agent_optimizer)



            # if np.random.rand()<=epsilon:
            #     new_action=np.random.randint(0,env.action_space.n)
            # else:
            #
            #     new_action = np.argmax(Q_table[new_observation,:])
            #
            # Q_table[observation,action]+=alpha*(reward+gamma*np.max(Q_table[new_observation,:])-Q_table[observation,action])
            #
            observation=new_observation
            # action=new_action

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                episodes_rewards[i_episode]=total_reward
                last_100_mean[i_episode]=np.mean(episodes_rewards[max(0,i_episode-100):i_episode])
                break

        if i_episode%50:
            print('episode',i_episode)









    #print(','.join([map_to_action(x) for x in optim]))
    plt.figure(figsize=(15,8))
    plt.plot(episodes_rewards)
    plt.axhline(y=0)
    plt.title('Total reward for each episode')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig('plot_total_reward_per_episode_deb.png', format = 'png')
    #
    plt.figure(figsize=(15,8))
    plt.plot(last_100_mean)
    plt.axhline(y=0)
    plt.title('Average reward last 100 episodes')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig('plot_avg_reward_last_100_deb.png', format = 'png')

    #torch.save(agent.state_dict(),  "agent")
    #torch.save(agent_optimizer.state_dict(), "agent_optimizer")

    env.close()

#sarsa_agent('SFFFHFFFFFFFFFFG',1.0,0.25,0.29,14697,741684)
#sarsa_agent('SFFFFHFFFFFFFFFFFFFFFFFFG',0.91,0.12,0.13,42271,983459)
Q=sarsa_agent(gamma=0.9,alpha=0.2,epsilon=0.4,n_episodes=500,seed=722060)
