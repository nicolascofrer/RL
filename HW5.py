import numpy as np
import itertools as it
# atEstabblishment=np.array([[1,1], [1,0], [0,1], [1,1], [0,0], [1,0], [1,1]])
# fightOccurred = np.array([0, 1, 0, 0, 0, 1, 0])

#print(atEstabblishment.shape)
def solve_KWIK(atEstabblishment,fightOccurred):

    atEstabblishment=atEstabblishment.replace('{','[')
    atEstabblishment=atEstabblishment.replace('}',']')
    print(atEstabblishment)
    atEstabblishment=np.array(eval(atEstabblishment))

    fightOccurred=fightOccurred.replace('{','[')
    fightOccurred=fightOccurred.replace('}',']')
    fightOccurred=np.array(eval(fightOccurred))

    N_patrons=atEstabblishment.shape[1]
    id_instigator_h=list(range(N_patrons))
    id_peace_maker_h=list(range(N_patrons))

    combinations_I=list(it.product(np.array([0,1]),repeat=N_patrons))
    combinations_P=list(it.product(np.array([0,1]),repeat=N_patrons))
    valid_I=np.ones(len(combinations_I))
    valid_P=np.ones(len(combinations_P))

    #print(combinations_I)


    for h_i in range(len(combinations_I)):
        #print('h is:',combinations_I[h_i])
        if np.sum(combinations_I[h_i])!=1:
            valid_I[h_i]=0

    for h_p in range(len(combinations_P)):
        #print('h is:',combinations_P[h_p])
        if np.sum(combinations_P[h_p])!=1:
            valid_P[h_p]=0


    new_H_I=list(it.compress(combinations_I,valid_I))
    new_H_P=list(it.compress(combinations_P,valid_P))

    H=list(it.product(new_H_I,new_H_P))
    #print('H',H)
    valid_H=np.ones(len(H))
    #print('valid_H',valid_H)
    for h in range(len(H)):
        #print(H[h])
        #print(np.array(H[h]))

        if (np.sum(np.array(H[h]),axis=0)>1).any():
            valid_H[h]=0

    #print(list(it.compress(H,valid_H)))

    H=list(it.compress(H,valid_H))
    #print('number of H', len(H))

        #NC: prediction using h in H:
    output=np.zeros(len(atEstabblishment))
    for i in range(len(fightOccurred)):
        #print('number of H', len(H))
        prediction=np.zeros(len(H))

        #print('episode',i)
        for h in range(len(H)):
            #print('h',H[h])
            #print('atEstabblishment[i]',atEstabblishment[i])
            instigator_present=np.sum(np.array(list(it.compress(list(H[h][0]),atEstabblishment[i]))))
            peace_maker_present=np.sum(np.array(list(it.compress(list(H[h][1]),atEstabblishment[i]))))

            if instigator_present and not peace_maker_present:
                prediction[h]=1
            #print('prediction[h]',prediction[h])

        prediction_set=set(prediction)
        #print('prediction_set',prediction_set)
        output[i]=-1
        if len(prediction_set)==1:
            output[i]=prediction_set.pop()

        else:

            compatible_H=prediction==fightOccurred[i]
            H=list(it.compress(H,compatible_H))

    print('output',str(output).replace(" ","").replace(" "," ").replace(".",","))
    return output

#solve_KWIK(np.array([[1,1], [1,0], [0,1], [1,1], [0,0], [1,0], [1,1]]),np.array([0, 1, 0, 0, 0, 1, 0]))
#solve_KWIK(np.array([[1,1],[1,1],[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]]),np.array([0,0,0,0,1,0,0,0]))
#solve_KWIK(np.array([[1,1,1],[0,0,1],[1,1,1],[1,1,1],[1,1,1],[0,1,1],[0,0,1],[0,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[0,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]),np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
#solve_KWIK(np.array([[0,0,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,0,1],[0,0,0,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,0,0,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,1,1],[0,0,0,1],[0,1,1,1],[1,1,1,1],[0,0,0,1],[1,1,1,1]]),np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
solve_KWIK(' {{1,1},{0,1},{0,1},{0,1},{0,1},{1,1},{0,1},{0,1}}',
'{0,1,1,1,1,0,1,1}')
