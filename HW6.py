from cvxopt import matrix, solvers

# A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
# b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
# c = matrix([ 2.0, 1.0 ])
# sol=solvers.lp(c,A,b)
#
# print('sol\n', sol['x'])


def solve_NE(game):
    A=matrix([1.0,1.0,1.0,0.0,0.0,
            -game[0][0],-game[1][0],-game[2][0],1.0,-1.0,
            -game[0][1],-game[1][1],-game[2][1],1.0,-1.0,
            -game[0][2],-game[1][2],-game[2][2],1.0,-1.0
            ],(5,4))
    print('A\n',A)
    b=matrix([ 0.0,0.0,0.0,1.0,-1.0 ])
    print('b\n',b)
    c = matrix([ -1.0,0.0,0.0,0.0 ])
    print('c\n',c)
    sol=solvers.lp(c,A,b)
    print('sol\n', sol['x'])


solve_NE(
[[0.0, 0.55, -2.28], [-0.55, 0.0, 2.78], [2.28, -2.78, 0.0]]
)
