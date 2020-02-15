import numpy as np
import numpy.random as rd
r=[7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6]

N=len(r)
lamb=1
gamma=1
#global e
#e=np.zeros(N)
probState=0.81
V=np.zeros(N)


def updateV(s_old,s_new,r):
    global e, steps,V,alphaT
    e[s_old]+=1
    print('e is ',e)
    steps+=1
    for i in range(N):
        print('Updating state ',i)
        V[i]=V[i]+(r+gamma*V[s_new]-V[s_old])*alphaT*e[i]

    alphaT=1/steps
    e=e*gamma
    print('new V is ',V)


episodes=0
steps=1
alphaT=1
while episodes<100:
    print('episodes ',episodes)
    e=np.zeros(N)
    #steps=1
    #alphaT=1
    s_old=0
    if rd.uniform()<probState:
        print('State is 1')
        s_new=1
        updateV(s_old,s_new,r[0])
        s_old=s_new
        s_new=3
        updateV(s_old,s_new,r[2])

        for i in range(4,7):
            print('State is ',i)

            s_old=s_new
            s_new=i
            updateV(s_old,s_new,r[i])

    else:
        print('State is 2')
        s_new=2
        updateV(s_old,s_new,r[1])
        s_old=s_new
        s_new=3
        updateV(s_old,s_new,r[3])

        for i in range(4,7):
            print('State is ',i)

            s_old=s_new
            s_new=i
            updateV(s_old,s_new,r[i])
    episodes+=1
