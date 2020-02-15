import numpy as np
def DieN(N,isBadSide):
    isBadSide=np.array(isBadSide)
    tol=1e-4
    delta=0.1
    #print('N ',N)
    #print('isBadSide ',isBadSide)
    maxS=100
    V=np.random.rand(maxS)
    policy=np.zeros(maxS)
    #print('V ',V)
    while delta>tol:
        delta=0
        for s in range(len(V)):
            #print('state is ',s)
            continuation=0
            v=V[s]
            winning_draws=np.argwhere(isBadSide==0)
            prob_win=0
            for win in winning_draws:
                #print('win is ',int(win)+1)
                max_winnings_cap= min(int(win)+1+s,maxS-1)
                continuation+=(1/N)*(int(win)+1+V[max_winnings_cap])
                prob_win+=(1/N)
            continuation+=-(1-prob_win)*s
            policy[s]=continuation>0
            #print('continuation ',continuation)
            #print('1-prob_win ',1-prob_win)
            V[s]=max(continuation,0)
            #print('abs(V[s]-v) ',abs(V[s]-v))
            delta=max(delta,abs(V[s]-v))
            #print('expected state value ',V[s])
            #print('expected winnings ',V[s]-s)
            #print('Delta ', delta)

    #print('policy ',policy)
    print('Optimal value initial state', V[0])
    #print('Optimal value 4 state', V[4])
    #print('Optimal value 5 state', V[5])
    #print('Optimal value 6 state', V[6])


    return policy


#DieN(6,[1, 1, 1, 0, 0, 0])

#DieN(21,[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
DieN(18,[0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1])
DieN(17,[0,0,1,0,1,0,1,1,0,1,1,1,1,1,0,1,0])
DieN(2,[0,1])
DieN(21,[0,0,0,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1])
DieN(26,[0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,0,0,1,0])
DieN(18,[0,0,0,0,1,0,0,0,1,1,1,1,0,1,1,1,1,0])
DieN(4,[0,1,1,0])
DieN(19,[0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0])
DieN(21,[0,1,1,1,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,0,0])
DieN(4,[0,0,0,1])
