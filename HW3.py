import gym
import numpy as np
import matplotlib.pyplot as plt

def sarsa_agent(amap,
gamma,
alpha,
epsilon,
n_episodes,
seed):

    input_string=amap
    N_episodes=n_episodes
    random_seed=seed
    #env = gym.make('FrozenLake-v0')
    #input_string='SFFFHFFFFFFFFFFG'
    map=np.asarray(input_string,dtype='c')
    map=map.reshape(int(np.sqrt(len(input_string))),int(np.sqrt(len(input_string))))
    print(map.shape)

    env = gym.make('FrozenLake-v0', desc=map).unwrapped
    #env.reset()
    #env = gym.make('CartPole-v0')
    print('env.action_space',env.action_space)
    print('env.observation_space',env.observation_space)

    print((env.observation_space.n))
    #alpha=0.25
    #gamma=1.0
    #random_seed=741684
    #epsilon=0.29
    #N_episodes=14697

    Q_table=np.zeros((env.observation_space.n,env.action_space.n))

    np.random.seed(random_seed)
    env.seed(random_seed)
    for i_episode in range(N_episodes):
        observation = env.reset()
        if np.random.rand()<=epsilon:

            action=np.random.randint(0,env.action_space.n)
        else:


            action = np.argmax(Q_table[observation,:])

        for t in range(5000):
            #env.render()
            #print(observation)






            new_observation, reward, done, info = env.step(action)



            if np.random.rand()<=epsilon:
                new_action=np.random.randint(0,env.action_space.n)
            else:

                new_action = np.argmax(Q_table[new_observation,:])

            Q_table[observation,action]+=alpha*(reward+gamma*Q_table[new_observation,new_action]-Q_table[observation,action])

            observation=new_observation
            action=new_action

            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break



    print(Q_table)
    def map_to_action(action_int):
        if action_int==0:
            return '<'
        elif action_int==1:
            return 'v'
        elif action_int==2:
            return '>'
        elif action_int==3:
            return '^'

    optim=np.argmax(Q_table,axis=1)
    print(','.join([map_to_action(x) for x in optim]))
    # plt.figure(figsize=(15,8))
    # plt.plot(episode_rewards)
    # plt.title('Total reward for each episode')
    # plt.ylabel('Total reward')
    # plt.xlabel('episode')
    # plt.savefig('plot_total_reward_per_episode.png', format = 'png')
    #

    env.close()

#sarsa_agent('SFFFHFFFFFFFFFFG',1.0,0.25,0.29,14697,741684)
#sarsa_agent('SFFFFHFFFFFFFFFFFFFFFFFFG',0.91,0.12,0.13,42271,983459)
sarsa_agent(amap='SFFG',
gamma=0.94,
alpha=0.23,
epsilon=0.2,
n_episodes=45890,
seed=722060)
