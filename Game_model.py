# Stochastic game model, consists of payoff function matrices, Q-value matrices initialization, game transition model, etc.
# A use case of system-of-systems life cycle optimization

## Number of agents = 2; Number of states for agent0 = 4, agent1 = 4 (including terminal states)
## Possible number of stage games = 9
## Termination criteria during learning: one or both agents reach terminal state(s)

import numpy as np

# Defining payoff function matrices correspond to the stage games
r_game_0_0 = [
    np.array([[0,0], [0,-2], [0,-4]]),
    np.array([[-5,5], [-5,3], [-5,1]]),
]
# r_game_0_0 = [
#     np.array([[0,0], [0,-2], [0,-4]]),
#     np.array([[-5,5], [-5,3], [-5,1]]),
# ]
r_game_1_0 = [
    np.array([[0,0], [0,-2], [0,-4]]),
    np.array([[6,3], [6,1], [6,-1]]),
]
r_game_2_0 = [
    np.array([[6,3], [6,1], [6,-1]]),
    np.array([[-5,5], [-5,3], [-5,1]]),
    np.array([[0,0], [0,-2], [0,-4]]),
]
r_game_0_1 = [
    np.array([[0,0], [0,-2], [0,0]]),
    np.array([[-8,8], [-8,6], [-8,8]]),
]
r_game_1_1 = [
    np.array([[0,0], [0,-2], [0,0]]),
    np.array([[10,4], [10,2], [10,0]]),
]
r_game_2_1 = [
    np.array([[10,4], [10,2], [10,4]]),
    np.array([[-8,8], [-8,6], [-8,8]]),
    np.array([[0,0], [0,-2], [0,0]]),
]
r_game_0_2 = [
    np.array([[0,0]]),
    np.array([[-10,-10]]),
]
r_game_1_2 = [
    np.array([[0,0]]),
    np.array([[15,7]]),
]
r_game_2_2 = [
    np.array([[15,7]]),
    np.array([[-10,10]]),
    np.array([[0,0]]),
]


# Initializing Q-value matrices correspond to the stage games
# Q_game_0_0 = [
#     np.zeros((3, 2)),
#     np.zeros((3, 2)),
# ]
# Trying with different initial values
Q_game_0_0 = [
    np.ones((3,2)),
    np.ones((3,2)),
]
Q_game_1_0 = [
    np.ones((3, 2)),
    np.ones((3, 2)),
]
Q_game_2_0 = [
    np.zeros((3, 2)),
    np.zeros((3, 2)),
    np.zeros((3, 2)),
]
# Q_game_2_0 = [
#     np.array([[10,1], [1,0], [0,1]]),
#     np.array([[1,0], [1,0], [0,1]]),
#     np.array([[0,1], [1,0], [0,1]]),
# ]
Q_game_0_1 = [
    np.zeros((3, 2)),
    np.zeros((3, 2)),
]
Q_game_1_1 = [
    np.zeros((3, 2)),
    np.zeros((3, 2)),
]
Q_game_2_1 = [
    np.zeros((3, 2)),
    np.zeros((3, 2)),
    np.zeros((3, 2)),
]
Q_game_0_2 = [
    np.zeros((1, 2)),
    np.zeros((1, 2)),
]
Q_game_1_2 = [
    np.zeros((1, 2)),
    np.zeros((1, 2)),
]
Q_game_2_2 = [
    np.zeros((1, 2)),
    np.zeros((1, 2)),
    np.zeros((1, 2)),
]

n_act0_game_0_0 = len(Q_game_0_0)
n_act1_game_0_0 = len(Q_game_0_0[0])
n_act0_game_1_0 = len(Q_game_1_0)
n_act1_game_1_0 = len(Q_game_1_0[0])
n_act0_game_2_0 = len(Q_game_2_0)
n_act1_game_2_0 = len(Q_game_2_0[0])
n_act0_game_0_1 = len(Q_game_0_1)
n_act1_game_0_1 = len(Q_game_0_1[0])
n_act0_game_1_1 = len(Q_game_1_1)
n_act1_game_1_1 = len(Q_game_1_1[0])
n_act0_game_2_1 = len(Q_game_2_1)
n_act1_game_2_1 = len(Q_game_2_1[0])
n_act0_game_0_2 = len(Q_game_0_2)
n_act1_game_0_2 = len(Q_game_0_2[0])
n_act0_game_1_2 = len(Q_game_1_2)
n_act1_game_1_2 = len(Q_game_1_2[0])
n_act0_game_2_2 = len(Q_game_2_2)
n_act1_game_2_2 = len(Q_game_2_2[0])