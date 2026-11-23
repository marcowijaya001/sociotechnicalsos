# A script for modeling a stochastic game and solving it using Nash Q-learning
# A use case of system-of-systems life cycle optimization
import numpy as np
import matplotlib.pyplot as plt
import time
import pygambit as gbt
from Game_model import *
from Nash_equilibrium import NashEquilibriumFinder, GlobalOptimumPoint, IsSaddlePoint
from Nash_equilibrium_generic import NashEquilibriumFinder

beta = 0.85 # discount factor

# Initializing all Q-values

Q_game_0_0_new = [np.zeros_like(a) for a in Q_game_0_0]
Q_value_0_0_0 = []
Q_value_0_0_1 = []
Q_value_0_0_2 = []
Q_value_0_0_3 = []
Q_value_0_0_4 = []
Q_value_0_0_5 = []
Q_value_0_1_0 = []
Q_value_0_1_1 = []
Q_value_0_1_2 = []
Q_value_0_1_3 = []
Q_value_0_1_4 = []
Q_value_0_1_5 = []
res_Q_value_0_0_0 = []

Q_game_0_1_new = [np.zeros_like(a) for a in Q_game_0_1]
Q_value_0_1 = []
Q_game_0_2_new = [np.zeros_like(a) for a in Q_game_0_2]
Q_value_0_2 = []
Q_game_1_0_new = [np.zeros_like(a) for a in Q_game_1_0]
Q_value_1_0 = []
Q_game_1_1_new = [np.zeros_like(a) for a in Q_game_1_1]
Q_value_1_1 = []
Q_game_1_2_new = [np.zeros_like(a) for a in Q_game_1_2]
Q_value_1_2 = []

Q_game_2_0_new = [np.zeros_like(a) for a in Q_game_2_0]
Q_value_2_0_0 = []
Q_value_2_0_1 = []
Q_value_2_0_2 = []
Q_value_2_0_3 = []
Q_value_2_0_4 = []
Q_value_2_0_5 = []
Q_value_2_0_6 = []
Q_value_2_0_7 = []
Q_value_2_0_8 = []
Q_value_2_0_9 = []
res_Q_value_2_0_0 = []

Q_game_2_1_new = [np.zeros_like(a) for a in Q_game_2_1]
Q_value_2_1 = []
Q_game_2_2_new = [np.zeros_like(a) for a in Q_game_2_2]
Q_value_2_2 = []

# Initializing all Nash Q-values from initial Q_game

NashQ_0_0_list, _, _ = NashEquilibriumFinder(Q_game_0_0)
NashQ_0_1_list, _, _ = NashEquilibriumFinder(Q_game_0_1)
NashQ_0_2_list, _, _ = NashEquilibriumFinder(Q_game_0_2)
NashQ_1_0_list, _, _ = NashEquilibriumFinder(Q_game_1_0)
NashQ_1_1_list, _, _ = NashEquilibriumFinder(Q_game_1_1)
NashQ_1_2_list, _, _ = NashEquilibriumFinder(Q_game_1_2)
NashQ_2_0_list, _, _ = NashEquilibriumFinder(Q_game_2_0)
NashQ_2_1_list, _, _ = NashEquilibriumFinder(Q_game_2_1)
NashQ_2_2_list, _, _ = NashEquilibriumFinder(Q_game_2_2)

# Iterating all Q-values using Nash Q-learning algorithm
n_iteration = 36

start_time = time.time()


for n in range(n_iteration):

    # --- Game 00, 6 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s0, s0)
    NashQ_0_0_list, _, _ = NashEquilibriumFinder(Q_game_0_0)
    print(NashQ_0_0_list)
    ### Agent 0 update
    Q_game_0_0_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_0_0[0][0][0] + 1 / (n + 1) * (r_game_0_0[0][0][0] + beta * NashQ_0_0_list[0][0][0]) # Q: to add stochasticity, the Nash Q-values simply come from the summation multiple future games with certain probabilities?
    Q_game_0_0[0][0][0] = Q_game_0_0_new[0][0][0]
    ### Agent 1 update
    Q_game_0_0_new[0][0][1] = (1 - 1 / (n +1)) * Q_game_0_0[0][0][1] + 1 / (n + 1) * (r_game_0_0[0][0][1] + beta * NashQ_0_0_list[0][0][1])
    Q_game_0_0[0][0][1] = Q_game_0_0_new[0][0][1]

    ## Action pair (a0, a1) -> final state (s0, s1)
    NashQ_0_1_list, _, _ = NashEquilibriumFinder(Q_game_0_1)
    ### Agent 0 update
    Q_game_0_0_new[0][1][0] = (1 - 1 / (n + 1)) * Q_game_0_0[0][1][0] + 1 / (n + 1) * (r_game_0_0[0][1][0] + beta * NashQ_0_1_list[0][0][0])
    Q_game_0_0[0][1][0] = Q_game_0_0_new[0][1][0]
    ### Agent 1 update
    Q_game_0_0_new[0][1][1] = (1 - 1 / (n + 1)) * Q_game_0_0[0][1][1] + 1 / (n + 1) * (r_game_0_0[0][1][1] + beta * NashQ_0_1_list[0][0][1])
    Q_game_0_0[0][1][1] = Q_game_0_0_new[0][1][1]

    ## Action pair (a0, a2) -> final state (s0, s2)
    NashQ_0_2_list, _, _ = NashEquilibriumFinder(Q_game_0_2)
    ### Agent 0 update
    Q_game_0_0_new[0][2][0] = (1 - 1 / (n + 1)) * Q_game_0_0[0][2][0] + 1 / (n + 1) * (r_game_0_0[0][2][0] + beta * NashQ_0_2_list[0][0][0])
    Q_game_0_0[0][2][0] = Q_game_0_0_new[0][2][0]
    ### Agent 1 update
    Q_game_0_0_new[0][2][1] = (1 - 1 / (n + 1)) * Q_game_0_0[0][2][1] + 1 / (n + 1) * (r_game_0_0[0][2][1] + beta * NashQ_0_2_list[0][0][1])
    Q_game_0_0[0][2][1] = Q_game_0_0_new[0][2][1]

    ## Action pair (a1, a0) -> final state (s1, s0)
    NashQ_1_0_list, _, _ = NashEquilibriumFinder(Q_game_1_0)
    ### Agent 0 update
    # NashQ = 10 + 0.1*n  # to be replaced with Nash Q-value from Nash equilibrium function, based on which game next when a pair of actions is selected
    Q_game_0_0_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_0_0[1][0][0] + 1 / (n + 1) * (r_game_0_0[1][0][0] + beta * NashQ_1_0_list[0][0][0])
    Q_game_0_0[1][0][0] = Q_game_0_0_new[1][0][0]
    ### Agent 1  update
    # NashQ1 = 10 + 0.1*n # for testing, to be replaces with actual iteration
    Q_game_0_0_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_0_0[1][0][1] + 1 / (n + 1) * (r_game_0_0[1][0][1] + beta * NashQ_1_0_list[0][0][1])
    Q_game_0_0[1][0][1] = Q_game_0_0_new[1][0][1]

    ## Action pair (a1, a1) -> final state (s1, s1)
    NashQ_1_1_list, _, _ = NashEquilibriumFinder(Q_game_1_1)
    ### Agent 0 update
    Q_game_0_0_new[1][1][0] = (1 - 1 / (n + 1)) * Q_game_0_0[1][1][0] + 1 / (n + 1) * (r_game_0_0[1][1][0] + beta * NashQ_1_1_list[0][0][0])
    Q_game_0_0[1][1][0] = Q_game_0_0_new[1][1][0]
    ### Agent 1 update
    Q_game_0_0_new[1][1][1] = (1 - 1 / (n + 1)) * Q_game_0_0[1][1][1] + 1 / (n + 1) * (r_game_0_0[1][1][1] + beta * NashQ_0_2_list[0][0][1])
    Q_game_0_0[1][1][1] = Q_game_0_0_new[1][1][1]

    ## Action pair (a1, a2) -> final state (s1, s2)
    NashQ_1_2_list, _, _ = NashEquilibriumFinder(Q_game_1_2)
    ### Agent 0 update
    Q_game_0_0_new[1][2][0] = (1 - 1 / (n + 1)) * Q_game_0_0[1][2][0] + 1 / (n + 1) * (r_game_0_0[1][2][0] + beta * NashQ_1_2_list[0][0][0])
    Q_game_0_0[1][2][0] = Q_game_0_0_new[1][2][0]
    ### Agent 1 update
    Q_game_0_0_new[1][2][1] = (1 - 1 / (n + 1)) * Q_game_0_0[1][2][1] + 1 / (n + 1) * (r_game_0_0[1][2][1] + beta * NashQ_1_2_list[0][0][1])
    Q_game_0_0[1][2][1] = Q_game_0_0_new[1][2][1]

    Q_value_0_0_0.append(Q_game_0_0[0][0][0])
    Q_value_0_0_1.append(Q_game_0_0[0][0][1])
    Q_value_0_0_2.append(Q_game_0_0[0][1][0])
    Q_value_0_0_3.append(Q_game_0_0[0][1][1])
    Q_value_0_0_4.append(Q_game_0_0[0][2][0])
    Q_value_0_0_5.append(Q_game_0_0[0][2][1])
    Q_value_0_1_0.append(Q_game_0_0[1][0][0])
    Q_value_0_1_1.append(Q_game_0_0[1][0][1])
    Q_value_0_1_2.append(Q_game_0_0[1][1][0])
    Q_value_0_1_3.append(Q_game_0_0[1][1][1])
    Q_value_0_1_4.append(Q_game_0_0[1][2][0])
    Q_value_0_1_5.append(Q_game_0_0[1][2][1])

    res_Q_value_0_0_0.append((Q_value_0_0_2[n] - Q_value_0_0_2[n-1]) / Q_value_0_0_1[n-1])

    print("\n --- Iteration ", n, "--- \n")
    print("Q_game_0_0: \n", Q_game_0_0)
    # print("Residual of Q-value: ", res_Q_value_0_0_0)


    # --- Game 01, 6 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s0, s1)
    NashQ_0_1_list, _, _ = NashEquilibriumFinder(Q_game_0_1)
    ### Agent 0 update
    Q_game_0_1_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_0_1[0][0][0] + 1 / (n + 1) * (r_game_0_1[0][0][0] + beta * NashQ_0_1_list[0][0][0])
    Q_game_0_1[0][0][0] = Q_game_0_0_new[0][0][0]
    ### Agent 1 update
    Q_game_0_1_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_0_1[0][0][1] + 1 / (n + 1) * (r_game_0_1[0][0][1] + beta * NashQ_0_1_list[0][0][1])
    Q_game_0_1[0][0][1] = Q_game_0_0_new[0][0][1]

    ## Action pair (a0,a1) -> final state (s0, s2)
    NashQ_0_2_list, _, _ = NashEquilibriumFinder(Q_game_0_2)
    ### Agent 0 update
    Q_game_0_1_new[0][1][0] = (1 - 1 / (n + 1)) * Q_game_0_1[0][1][0] + 1 / (n + 1) * (r_game_0_1[0][1][0] + beta * NashQ_0_2_list[0][0][0])
    Q_game_0_1[0][1][0] = Q_game_0_1_new[0][1][0]
    ### Agent 1 update
    Q_game_0_1_new[0][1][1] = (1 - 1 / (n + 1)) * Q_game_0_1[0][1][1] + 1 / (n + 1) * (r_game_0_1[0][1][1] + beta * NashQ_0_2_list[0][0][1])
    Q_game_0_1[0][1][1] = Q_game_0_1_new[0][1][1]

    ## Action pair (a0,a2) -> final state (s0, s3: END); Nash Q-value for transition to END is assumed to be 0
    ### Agent 0 update
    Q_game_0_1_new[0][2][0] = (1 - 1 / (n + 1)) * Q_game_0_1[0][2][0] + 1 / (n + 1) * (r_game_0_1[0][2][0] + beta * 0)
    Q_game_0_1[0][2][0] = Q_game_0_1_new[0][2][0]
    ### Agent 1 update
    Q_game_0_1_new[0][2][1] = (1 - 1 / (n + 1)) * Q_game_0_1[0][2][1] + 1 / (n + 1) * (r_game_0_1[0][2][1] + beta * 0)
    Q_game_0_1[0][2][1] = Q_game_0_1_new[0][2][1]

    ## Action pair (a1,a0) -> final state (s1, s1)
    NashQ_1_1_list, _, _ = NashEquilibriumFinder(Q_game_1_1)
    ### Agent 0 update
    Q_game_0_1_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_0_1[1][0][0] + 1 / (n + 1) * (r_game_0_1[1][0][0] + beta * NashQ_1_1_list[0][0][0])
    Q_game_0_1[1][0][0] = Q_game_0_0_new[1][0][0]
    ### Agent 1 update
    Q_game_0_1_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_0_1[1][0][1] + 1 / (n + 1) * (r_game_0_1[1][0][1] + beta * NashQ_1_1_list[0][0][1])
    Q_game_0_1[1][0][1] = Q_game_0_0_new[1][0][1]

    ## Action pair (a1,a1) -> final state (s1, s2)
    NashQ_1_2_list, _, _ = NashEquilibriumFinder(Q_game_1_2)
    ### Agent 0 update
    Q_game_0_1_new[1][1][0] = (1 - 1 / (n + 1)) * Q_game_0_1[1][1][0] + 1 / (n + 1) * (r_game_0_1[1][1][0] + beta * NashQ_1_2_list[0][0][0])
    Q_game_0_1[1][1][0] = Q_game_0_0_new[1][1][0]
    ### Agent 1 update
    Q_game_0_1_new[1][1][1] = (1 - 1 / (n + 1)) * Q_game_0_1[1][1][1] + 1 / (n + 1) * (r_game_0_1[1][1][1] + beta * NashQ_1_2_list[0][0][1])
    Q_game_0_1[1][1][1] = Q_game_0_0_new[1][1][1]

    ## Action pair (a1,a2) -> final state (s1, s3: END)
    ### Agent 0 update
    Q_game_0_1_new[1][2][0] = (1 - 1 / (n + 1)) * Q_game_0_1[1][2][0] + 1 / (n + 1) * (r_game_0_1[1][2][0] + beta * 0)
    Q_game_0_1[1][2][0] = Q_game_0_0_new[1][2][0]
    ### Agent 1 update
    Q_game_0_1_new[1][2][1] = (1 - 1 / (n + 1)) * Q_game_0_1[1][2][1] + 1 / (n + 1) * (r_game_0_1[1][2][1] + beta * 0)
    Q_game_0_1[1][2][1] = Q_game_0_0_new[1][2][1]

    # print("Q_game_0_1: \n",Q_game_0_1)
    # Q_value_0_1.append(Q_game_0_1[0][2][1])


    # --- Game 02, 2 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s0, s3: END)
    ### Agent 0 update
    Q_game_0_2_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_0_2[0][0][0] + 1 / (n + 1) * (r_game_0_2[0][0][0] + beta * 0)
    Q_game_0_2[0][0][0] = Q_game_0_2_new[0][0][0]
    ### Agent 1 update
    Q_game_0_2_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_0_2[0][0][1] + 1 / (n + 1) * (r_game_0_2[0][0][1] + beta * 0)
    Q_game_0_2[0][0][1] = Q_game_0_2_new[0][0][1]

    ## Action pair (a1,a0) -> final state (s0, s3: END)
    ### Agent 0 update
    Q_game_0_2_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_0_2[1][0][0] + 1 / (n + 1) * (r_game_0_2[1][0][0] + beta * 0)
    Q_game_0_2[1][0][0] = Q_game_0_2_new[1][0][0]
    ### Agent 1 update
    Q_game_0_2_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_0_2[1][0][1] + 1 / (n + 1) * (r_game_0_2[1][0][1] + beta * 0)
    Q_game_0_2[1][0][1] = Q_game_0_2_new[1][0][1]

    # print("Q_game_0_2: \n",Q_game_0_2)


    # --- Game 10, 6 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s1, s0)
    NashQ_1_0_list, _, _ = NashEquilibriumFinder(Q_game_1_0)
    ### Agent 0 update
    Q_game_1_0_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_1_0[0][0][0] + 1 / (n + 1) * (r_game_1_0[0][0][0] + beta * NashQ_1_0_list[0][0][0])
    Q_game_1_0[0][0][0] = Q_game_1_0_new[0][0][0]
    ### Agent 1 update
    Q_game_1_0_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_1_0[0][0][1] + 1 / (n + 1) * (r_game_1_0[0][0][1] + beta * NashQ_1_0_list[0][0][1])
    Q_game_1_0[0][0][1] = Q_game_1_0_new[0][0][1]

    ## Action pair (a0,a1) -> final state (s1, s1)
    NashQ_1_1_list, _, _ = NashEquilibriumFinder(Q_game_1_1)
    ### Agent 0 update
    Q_game_1_0_new[0][1][0] = (1 - 1 / (n + 1)) * Q_game_1_0[0][1][0] + 1 / (n + 1) * (r_game_1_0[0][1][0] + beta * NashQ_1_1_list[0][0][0])
    Q_game_1_0[0][1][0] = Q_game_1_0_new[0][1][0]
    ### Agent 1 update
    Q_game_1_0_new[0][1][1] = (1 - 1 / (n + 1)) * Q_game_1_0[0][1][1] + 1 / (n + 1) * (r_game_1_0[0][1][1] + beta * NashQ_1_1_list[0][0][1])
    Q_game_1_0[0][1][1] = Q_game_1_0_new[0][1][1]

    ## Action pair (a0,a2) -> final state (s1, s2)
    NashQ_1_2_list, _, _ = NashEquilibriumFinder(Q_game_1_2)
    ### Agent 0 update
    Q_game_1_0_new[0][2][0] = (1 - 1 / (n + 1)) * Q_game_1_0[0][2][0] + 1 / (n + 1) * (r_game_1_0[0][2][0] + beta * NashQ_1_2_list[0][0][0])
    Q_game_1_0[0][2][0] = Q_game_1_0_new[0][2][0]
    ### Agent 1 update
    Q_game_1_0_new[0][2][1] = (1 - 1 / (n + 1)) * Q_game_1_0[0][2][1] + 1 / (n + 1) * (r_game_1_0[0][2][1] + beta * NashQ_1_2_list[0][0][1])
    Q_game_1_0[0][2][1] = Q_game_1_0_new[0][2][1]

    ## Action pair (a1,a0) -> final state (s2, s0)
    NashQ_2_0_list, _, _ = NashEquilibriumFinder(Q_game_2_0)
    ### Agent 0 update
    Q_game_1_0_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_1_0[1][0][0] + 1 / (n + 1) * (r_game_1_0[1][0][0] + beta * NashQ_2_0_list[0][0][0])
    Q_game_1_0[1][0][0] = Q_game_1_0_new[1][0][0]
    ### Agent 1 update
    Q_game_1_0_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_1_0[1][0][1] + 1 / (n + 1) * (r_game_1_0[1][0][1] + beta * NashQ_2_0_list[0][0][1])
    Q_game_1_0[1][0][1] = Q_game_1_0_new[1][0][1]

    ## Action pair (a1,a1) -> final state (s2, s1)
    NashQ_2_1_list, _, _ = NashEquilibriumFinder(Q_game_2_1)
    ### Agent 0 update
    Q_game_1_0_new[1][1][0] = (1 - 1 / (n + 1)) * Q_game_1_0[1][1][0] + 1 / (n + 1) * (r_game_1_0[1][1][0] + beta * (0.8 * NashQ_2_1_list[0][0][0] + 0.2 * NashQ_2_2_list[0][0][0])) # there is stochasticity for state transition  here
    Q_game_1_0[1][1][0] = Q_game_1_0_new[1][1][0]
    ### Agent 1 update
    Q_game_1_0_new[1][1][1] = (1 - 1 / (n + 1)) * Q_game_1_0[1][1][1] + 1 / (n + 1) * (r_game_1_0[1][1][1] + beta * (0.8 * NashQ_2_1_list[0][0][1] + 0.2 * NashQ_2_2_list[0][0][1]))
    Q_game_1_0[1][1][1] = Q_game_1_0_new[1][1][1]

    ## Action pair (a1,a2) -> final state (s2, s2)
    NashQ_2_2_list, _, _ = NashEquilibriumFinder(Q_game_2_2)
    ### Agent 0 update
    Q_game_1_0_new[1][2][0] = (1 - 1 / (n + 1)) * Q_game_1_0[1][2][0] + 1 / (n + 1) * (r_game_1_0[1][2][0] + beta * NashQ_2_2_list[0][0][0])
    Q_game_1_0[1][2][0] = Q_game_1_0_new[1][2][0]
    ### Agent 1 update
    Q_game_1_0_new[1][2][1] = (1 - 1 / (n + 1)) * Q_game_1_0[1][2][1] + 1 / (n + 1) * (r_game_1_0[1][2][1] + beta * NashQ_2_2_list[0][0][1])
    Q_game_1_0[1][2][1] = Q_game_1_0_new[1][2][1]

    print("Q_game_1_0: \n",Q_game_1_0)


    # --- Game 11, 6 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s1, s1)
    NashQ_1_1_list, _, _ = NashEquilibriumFinder(Q_game_1_1)
    ### Agent 0 update
    Q_game_1_1_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_1_1[0][0][0] + 1 / (n + 1) * (r_game_1_1[0][0][0] + beta * NashQ_1_1_list[0][0][0])
    Q_game_1_1[0][0][0] = Q_game_1_1_new[0][0][0]
    ### Agent 1 update
    Q_game_1_1_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_1_1[0][0][1] + 1 / (n + 1) * (r_game_1_1[0][0][1] + beta * NashQ_1_1_list[0][0][1])
    Q_game_1_1[0][0][1] = Q_game_1_1_new[0][0][1]

    ## Action pair (a0,a1) -> final state (s1, s2)
    NashQ_1_2_list, _, _ = NashEquilibriumFinder(Q_game_1_2)
    ### Agent 0 update
    Q_game_1_1_new[0][1][0] = (1 - 1 / (n + 1)) * Q_game_1_1[0][1][0] + 1 / (n + 1) * (r_game_1_1[0][1][0] + beta * NashQ_1_2_list[0][0][0])
    Q_game_1_1[0][1][0] = Q_game_1_1_new[0][0][0]
    ### Agent 1 update
    Q_game_1_1_new[0][1][1] = (1 - 1 / (n + 1)) * Q_game_1_1[0][1][1] + 1 / (n + 1) * (r_game_1_1[0][1][1] + beta * NashQ_1_2_list[0][0][1])
    Q_game_1_1[0][1][1] = Q_game_1_1_new[0][1][1]

    ## Action pair (a0,a2) -> final state (s1, s3: END)
    ### Agent 0 update
    Q_game_1_1_new[0][2][0] = (1 - 1 / (n + 1)) * Q_game_1_1[0][2][0] + 1 / (n + 1) * (r_game_1_1[0][2][0] + beta * 0)
    Q_game_1_1[0][2][0] = Q_game_1_1_new[0][2][0]
    ### Agent 1 update
    Q_game_1_1_new[0][2][1] = (1 - 1 / (n + 1)) * Q_game_1_1[0][2][1] + 1 / (n + 1) * (r_game_1_1[0][2][1] + beta * 0)
    Q_game_1_1[0][2][1] = Q_game_1_1_new[0][2][1]

    ## Action pair (a1,a0) -> final state (s2, s1)
    NashQ_2_1_list, _, _ = NashEquilibriumFinder(Q_game_2_1)
    ### Agent 0 update
    Q_game_1_1_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_1_1[1][0][0] + 1 / (n + 1) * (r_game_1_1[1][0][0] + beta * NashQ_2_1_list[0][0][0])
    Q_game_1_1[1][0][0] = Q_game_1_1_new[1][0][0]
    ### Agent 1 update
    Q_game_1_1_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_1_1[1][0][1] + 1 / (n + 1) * (r_game_1_1[1][0][1] + beta * NashQ_2_1_list[0][0][1])
    Q_game_1_1[1][0][1] = Q_game_1_1_new[1][0][1]

    ## Action pair (a1,a1) -> final state (s2, s2)
    NashQ_2_2_list, _, _ = NashEquilibriumFinder(Q_game_2_2)
    ### Agent 0 update
    Q_game_1_1_new[1][1][0] = (1 - 1 / (n + 1)) * Q_game_1_1[1][1][0] + 1 / (n + 1) * (r_game_1_1[1][1][0] + beta * NashQ_2_2_list[0][0][0])
    Q_game_1_1[1][1][0] = Q_game_1_1_new[1][0][0]
    ### Agent 1 update
    Q_game_1_1_new[1][1][1] = (1 - 1 / (n + 1)) * Q_game_1_1[1][1][1] + 1 / (n + 1) * (r_game_1_1[1][1][1] + beta * NashQ_2_2_list[0][0][1])
    Q_game_1_1[1][1][1] = Q_game_1_1_new[1][1][1]

    ## Action pair (a1,a2) -> final state (s2, s3: END)
    ### Agent 0 update
    Q_game_1_1_new[1][2][0] = (1 - 1 / (n + 1)) * Q_game_1_1[1][2][0] + 1 / (n + 1) * (r_game_1_1[1][2][0] + beta * 0)
    Q_game_1_1[1][2][0] = Q_game_1_1_new[1][2][0]
    ### Agent 1 update
    Q_game_1_1_new[1][2][1] = (1 - 1 / (n + 1)) * Q_game_1_1[1][2][1] + 1 / (n + 1) * (r_game_1_1[1][2][1] + beta * 0)
    Q_game_1_1[1][2][1] = Q_game_1_1_new[1][2][1]

    # print("Q_game_1_1: \n",Q_game_1_1)


    # --- Game 12, 2 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s1, s3: END)
    ### Agent 0 update
    Q_game_1_2_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_1_2[0][0][0] + 1 / (n + 1) * (r_game_1_2[0][0][0] + beta * 0)
    Q_game_1_2[0][0][0] = Q_game_1_2_new[0][0][0]
    ### Agent 1 update
    Q_game_1_2_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_1_2[0][0][1] + 1 / (n + 1) * (r_game_1_2[0][0][1] + beta * 0)
    Q_game_1_2[0][0][1] = Q_game_1_2_new[0][0][1]

    ## Action pair (a1,a0) -> final state (s2, s3: END)
    ### Agent 0 update
    Q_game_1_2_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_1_2[1][0][0] + 1 / (n + 1) * (r_game_1_2[0][0][0] + beta * 0)
    Q_game_1_2[1][0][0] = Q_game_1_2_new[1][0][0]
    ### Agent 1 update
    Q_game_1_2_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_1_2[1][0][1] + 1 / (n + 1) * (r_game_1_2[0][0][1] + beta * 0)
    Q_game_1_2[1][0][1] = Q_game_1_2_new[1][0][1]

    # print("Q_game_1_2: \n", Q_game_1_2)


    # --- Game 20, 9 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s2, s0)
    NashQ_2_0_list, _, _ = NashEquilibriumFinder(Q_game_2_0)
    ### Agent 0 update
    Q_game_2_0_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_2_0[0][0][0] + 1 / (n + 1) * (r_game_2_0[0][0][0] + beta * NashQ_2_0_list[0][0][0])
    Q_game_2_0[0][0][0] = Q_game_2_0_new[0][0][0]
    ### Agent 1 update
    Q_game_2_0_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_2_0[0][0][1] + 1 / (n + 1) * (r_game_2_0[0][0][1] + beta * NashQ_2_0_list[0][0][1])
    Q_game_2_0[0][0][1] = Q_game_2_0_new[0][0][1]

    ## Action pair (a0,a1) -> final state (s2, s1)
    NashQ_2_1_list, _, _ = NashEquilibriumFinder(Q_game_2_1)
    ### Agent 0 update
    Q_game_2_0_new[0][1][0] = (1 - 1 / (n + 1)) * Q_game_2_0[0][1][0] + 1 / (n + 1) * (r_game_2_0[0][1][0] + beta * NashQ_2_1_list[0][0][0])
    Q_game_2_0[0][1][0] = Q_game_2_0_new[0][1][0]
    ### Agent 1 update
    Q_game_2_0_new[0][1][1] = (1 - 1 / (n + 1)) * Q_game_2_0[0][1][1] + 1 / (n + 1) * (r_game_2_0[0][1][1] + beta * NashQ_2_1_list[0][0][1])
    Q_game_2_0[0][1][1] = Q_game_2_0_new[0][1][1]

    ## Action pair (a0,a2) -> final state (s2, s2)
    NashQ_2_2_list, _, _ = NashEquilibriumFinder(Q_game_2_2)
    ### Agent 0 update
    Q_game_2_0_new[0][2][0] = (1 - 1 / (n + 1)) * Q_game_2_0[0][2][0] + 1 / (n + 1) * (r_game_2_0[0][2][0] + beta * NashQ_2_2_list[0][0][0])
    Q_game_2_0[0][2][0] = Q_game_2_0_new[0][2][0]
    ### Agent 1 update
    Q_game_2_0_new[0][2][1] = (1 - 1 / (n + 1)) * Q_game_2_0[0][2][1] + 1 / (n + 1) * (r_game_2_0[0][2][1] + beta * NashQ_2_2_list[0][0][1])
    Q_game_2_0[0][2][1] = Q_game_2_0_new[0][2][1]

    ## Action pair (a1,a0) -> final state (s1, s0)
    NashQ_1_0_list, _, _ = NashEquilibriumFinder(Q_game_1_0)
    ### Agent 0 update
    Q_game_2_0_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_2_0[1][0][0] + 1 / (n + 1) * (r_game_2_0[1][0][0] + beta * NashQ_1_0_list[0][0][0])
    Q_game_2_0[1][0][0] = Q_game_2_0_new[1][0][0]
    ### Agent 1 update
    Q_game_2_0_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_2_0[1][0][1] + 1 / (n + 1) * (r_game_2_0[1][0][1] + beta * NashQ_1_0_list[0][0][1])
    Q_game_2_0[1][0][1] = Q_game_2_0_new[1][0][1]

    ## Action pair (a1,a1) -> final state (s1, s1)
    NashQ_1_1_list, _, _ = NashEquilibriumFinder(Q_game_1_1)
    ### Agent 0 update
    Q_game_2_0_new[1][1][0] = (1 - 1 / (n + 1)) * Q_game_2_0[1][1][0] + 1 / (n + 1) * (r_game_2_0[1][1][0] + beta * NashQ_1_1_list[0][0][0])
    Q_game_2_0[1][1][0] = Q_game_2_0_new[1][1][0]
    ### Agent 1 update
    Q_game_2_0_new[1][1][1] = (1 - 1 / (n + 1)) * Q_game_2_0[1][1][1] + 1 / (n + 1) * (r_game_2_0[1][1][1] + beta * NashQ_1_1_list[0][0][1])
    Q_game_2_0[1][1][1] = Q_game_2_0_new[1][1][1]

    ## Action pair (a1,a2) -> final state (s1, s2)
    NashQ_1_2_list, _, _ = NashEquilibriumFinder(Q_game_1_2)
    ### Agent 0 update
    Q_game_2_0_new[1][2][0] = (1 - 1 / (n + 1)) * Q_game_2_0[1][2][0] + 1 / (n + 1) * (r_game_2_0[1][2][0] + beta * NashQ_1_2_list[0][0][0])
    Q_game_2_0[1][2][0] = Q_game_2_0_new[1][2][0]
    ### Agent 1 update
    Q_game_2_0_new[1][2][1] = (1 - 1 / (n + 1)) * Q_game_2_0[1][2][1] + 1 / (n + 1) * (r_game_2_0[1][2][1] + beta * NashQ_1_2_list[0][0][1])
    Q_game_2_0[1][2][1] = Q_game_2_0_new[1][2][1]

    ## Action pair (a2,a0) -> final state (s3, s0: END)
    ### Agent 0 update
    Q_game_2_0_new[2][0][0] = (1 - 1 / (n + 1)) * Q_game_2_0[2][0][0] + 1 / (n + 1) * (r_game_2_0[2][0][0] + beta * 0)
    Q_game_2_0[2][0][0] = Q_game_2_0_new[2][0][0]
    ### Agent 1 update
    Q_game_2_0_new[2][0][1] = (1 - 1 / (n + 1)) * Q_game_2_0[2][0][1] + 1 / (n + 1) * (r_game_2_0[2][0][1] + beta * 0)
    Q_game_2_0[2][0][1] = Q_game_2_0_new[2][0][1]

    ## Action pair (a2,a1) -> final state (s3, s1: END)
    ### Agent 0 update
    Q_game_2_0_new[2][1][0] = (1 - 1 / (n + 1)) * Q_game_2_0[2][1][0] + 1 / (n + 1) * (r_game_2_0[2][1][0] + beta * 0)
    Q_game_2_0[2][1][0] = Q_game_2_0_new[2][1][0]
    ### Agent 1 update
    Q_game_2_0_new[2][1][1] = (1 - 1 / (n + 1)) * Q_game_2_0[2][1][1] + 1 / (n + 1) * (r_game_2_0[2][1][1] + beta * 0)
    Q_game_2_0[2][1][1] = Q_game_2_0_new[2][1][1]

    ## Action pair (a2,a2) -> final state (s3, s2: END)
    ### Agent 0 update
    Q_game_2_0_new[2][2][0] = (1 - 1 / (n + 1)) * Q_game_2_0[2][2][0] + 1 / (n + 1) * (r_game_2_0[2][2][0] + beta * 0)
    Q_game_2_0[2][2][0] = Q_game_2_0_new[2][2][0]
    ### Agent 1 update
    Q_game_2_0_new[2][2][1] = (1 - 1 / (n + 1)) * Q_game_2_0[2][2][1] + 1 / (n + 1) * (r_game_2_0[2][2][1] + beta * 0)
    Q_game_2_0[2][2][1] = Q_game_2_0_new[2][2][1]

    Q_value_2_0_0.append(Q_game_2_0[0][0][0])
    Q_value_2_0_1.append(Q_game_2_0[0][0][1])
    Q_value_2_0_2.append(Q_game_2_0[0][1][0])
    Q_value_2_0_3.append(Q_game_2_0[0][1][1])
    Q_value_2_0_4.append(Q_game_2_0[0][2][0])
    Q_value_2_0_5.append(Q_game_2_0[0][2][1])
    Q_value_2_0_6.append(Q_game_2_0[1][0][0])
    Q_value_2_0_7.append(Q_game_2_0[1][0][1])
    Q_value_2_0_8.append(Q_game_2_0[1][1][0])
    Q_value_2_0_9.append(Q_game_2_0[1][1][1])

    res_Q_value_2_0_0.append((Q_value_2_0_0[n] - Q_value_2_0_0[n - 1]) / Q_value_2_0_0[n - 1])

    print("Q_game_2_0: \n", Q_game_2_0)


    # --- Game 21, 9 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s2, s1)
    NashQ_2_1_list, _, _ = NashEquilibriumFinder(Q_game_2_1)
    ### Agent 0 update
    Q_game_2_1_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_2_1[0][0][0] + 1 / (n + 1) * (r_game_2_1[0][0][0] + beta * NashQ_2_1_list[0][0][0])
    Q_game_2_1[0][0][0] = Q_game_2_1_new[0][0][0]
    ### Agent 1 update
    Q_game_2_1_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_2_1[0][0][1] + 1 / (n + 1) * (r_game_2_1[0][0][1] + beta * NashQ_2_1_list[0][0][1])
    Q_game_2_1[0][0][1] = Q_game_2_1_new[0][0][1]

    ## Action pair (a0,a1) -> final state (s2, s2)
    NashQ_2_2_list, _, _ = NashEquilibriumFinder(Q_game_2_2)
    ### Agent 0 update
    Q_game_2_1_new[0][1][0] = (1 - 1 / (n + 1)) * Q_game_2_1[0][1][0] + 1 / (n + 1) * (r_game_2_1[0][1][0] + beta * NashQ_2_2_list[0][0][0])
    Q_game_2_1[0][1][0] = Q_game_2_1_new[0][1][0]
    ### Agent 1 update
    Q_game_2_1_new[0][1][1] = (1 - 1 / (n + 1)) * Q_game_2_1[0][1][1] + 1 / (n + 1) * (r_game_2_1[0][1][1] + beta * NashQ_2_2_list[0][0][1])
    Q_game_2_1[0][1][1] = Q_game_2_1_new[0][1][1]

    ## Action pair (a0,a2) -> final state (s2, s3: END)
    ### Agent 0 update
    Q_game_2_1_new[0][2][0] = (1 - 1 / (n + 1)) * Q_game_2_1[0][2][0] + 1 / (n + 1) * (r_game_2_1[0][2][0] + beta * 0)
    Q_game_2_1[0][2][0] = Q_game_2_1_new[0][2][0]
    ### Agent 1 update
    Q_game_2_1_new[0][2][1] = (1 - 1 / (n + 1)) * Q_game_2_1[0][2][1] + 1 / (n + 1) * (r_game_2_1[0][2][1] + beta * 0)
    Q_game_2_1[0][2][1] = Q_game_2_1_new[0][2][1]

    ## Action pair (a1,a0) -> final state (s1, s1)
    NashQ_1_1_list, _, _ = NashEquilibriumFinder(Q_game_1_1)
    ### Agent 0 update
    Q_game_2_1_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_2_1[1][0][0] + 1 / (n + 1) * (r_game_2_1[1][0][0] + beta * NashQ_1_1_list[0][0][0])
    Q_game_2_1[1][0][0] = Q_game_2_1_new[1][0][0]
    ### Agent 1 update
    Q_game_2_1_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_2_1[1][0][1] + 1 / (n + 1) * (r_game_2_1[1][0][1] + beta * NashQ_1_1_list[0][0][1])
    Q_game_2_1[1][0][1] = Q_game_2_1_new[1][0][1]

    ## Action pair (a1,a1) -> final state (s1, s2)
    NashQ_1_2_list, _, _ = NashEquilibriumFinder(Q_game_1_2)
    ### Agent 0 update
    Q_game_2_1_new[1][1][0] = (1 - 1 / (n + 1)) * Q_game_2_1[1][1][0] + 1 / (n + 1) * (r_game_2_1[1][1][0] + beta * NashQ_1_2_list[0][0][0])
    Q_game_2_1[1][1][0] = Q_game_2_1_new[1][1][0]
    ### Agent 1 update
    Q_game_2_1_new[1][1][1] = (1 - 1 / (n + 1)) * Q_game_2_1[1][1][1] + 1 / (n + 1) * (r_game_2_1[1][1][1] + beta * NashQ_1_2_list[0][0][1])
    Q_game_2_1[1][1][1] = Q_game_2_1_new[1][1][1]

    ## Action pair (a1,a2) -> final state (s1, s3: END)
    ### Agent 0 update
    Q_game_2_1_new[1][2][0] = (1 - 1 / (n + 1)) * Q_game_2_1[1][2][0] + 1 / (n + 1) * (r_game_2_1[1][2][0] + beta * 0)
    Q_game_2_1[1][2][0] = Q_game_2_1_new[1][2][0]
    ### Agent 1 update
    Q_game_2_1_new[1][2][1] = (1 - 1 / (n + 1)) * Q_game_2_1[1][2][1] + 1 / (n + 1) * (r_game_2_1[1][2][1] + beta * 0)
    Q_game_2_1[1][2][1] = Q_game_2_1_new[1][2][1]

    ## Action pair (a2,a0) -> final state (s3, s1: END)
    ### Agent 0 update
    Q_game_2_1_new[2][0][0] = (1 - 1 / (n + 1)) * Q_game_2_1[2][0][0] + 1 / (n + 1) * (r_game_2_1[2][0][0] + beta * 0)
    Q_game_2_1[2][0][0] = Q_game_2_1_new[2][0][0]
    ### Agent 1 update
    Q_game_2_1_new[2][0][1] = (1 - 1 / (n + 1)) * Q_game_2_1[2][0][1] + 1 / (n + 1) * (r_game_2_1[2][0][1] + beta * 0)
    Q_game_2_1[2][0][1] = Q_game_2_1_new[2][0][1]

    ## Action pair (a2,a1) -> final state (s3, s2: END)
    ### Agent 0 update
    Q_game_2_1_new[2][1][0] = (1 - 1 / (n + 1)) * Q_game_2_1[2][1][0] + 1 / (n + 1) * (r_game_2_1[2][1][0] + beta * 0)
    Q_game_2_1[2][1][0] = Q_game_2_1_new[2][1][0]
    ### Agent 1 update
    Q_game_2_1_new[2][1][1] = (1 - 1 / (n + 1)) * Q_game_2_1[2][1][1] + 1 / (n + 1) * (r_game_2_1[2][1][1] + beta * 0)
    Q_game_2_1[2][1][1] = Q_game_2_1_new[2][1][1]

    ## Action pair (a2,a2) -> final state (s3, s3: END)
    ### Agent 0 update
    Q_game_2_1_new[2][2][0] = (1 - 1 / (n + 1)) * Q_game_2_1[2][2][0] + 1 / (n + 1) * (r_game_2_1[2][2][0] + beta * 0)
    Q_game_2_1[2][2][0] = Q_game_2_1_new[2][2][0]
    ### Agent 1 update
    Q_game_2_1_new[2][2][1] = (1 - 1 / (n + 1)) * Q_game_2_1[2][2][1] + 1 / (n + 1) * (r_game_2_1[2][2][1] + beta * 0)
    Q_game_2_1[2][2][1] = Q_game_2_1_new[2][2][1]

    print("Q_game_2_1: \n", Q_game_2_1)


    # --- Game 22, 3 action pairs, 2 agent for each action pair --- #

    ## Action pair (a0,a0) -> final state (s2, s3: END)
    ### Agent 0 update
    Q_game_2_2_new[0][0][0] = (1 - 1 / (n + 1)) * Q_game_2_2[0][0][0] + 1 / (n + 1) * (r_game_2_2[0][0][0] + beta * 0)
    Q_game_2_2[0][0][0] = Q_game_2_2_new[0][0][0]
    ### Agent 1 update
    Q_game_2_2_new[0][0][1] = (1 - 1 / (n + 1)) * Q_game_2_2[0][0][1] + 1 / (n + 1) * (r_game_2_2[0][0][1] + beta * 0)
    Q_game_2_2[0][0][1] = Q_game_2_2_new[0][0][1]

    ## Action pair (a1,a0) -> final state (s1, s3: END)
    ### Agent 0 update
    Q_game_2_2_new[1][0][0] = (1 - 1 / (n + 1)) * Q_game_2_2[1][0][0] + 1 / (n + 1) * (r_game_2_2[1][0][0] + beta * 0)
    Q_game_2_2[1][0][0] = Q_game_2_2_new[1][0][0]
    ### Agent 1 update
    Q_game_2_2_new[1][0][1] = (1 - 1 / (n + 1)) * Q_game_2_2[1][0][1] + 1 / (n + 1) * (r_game_2_2[1][0][1] + beta * 0)
    Q_game_2_2[1][0][1] = Q_game_2_2_new[1][0][1]

    ## Action pair (a2,a0) -> final state (s3, s3: END)
    ### Agent 0 update
    Q_game_2_2_new[2][0][0] = (1 - 1 / (n + 1)) * Q_game_2_2[2][0][0] + 1 / (n + 1) * (r_game_2_2[2][0][0] + beta * 0)
    Q_game_2_2[2][0][0] = Q_game_2_2_new[2][0][0]
    ### Agent 1 update
    Q_game_2_2_new[2][0][1] = (1 - 1 / (n + 1)) * Q_game_2_2[2][0][1] + 1 / (n + 1) * (r_game_2_2[2][0][1] + beta * 0)
    Q_game_2_2[2][0][1] = Q_game_2_2_new[2][0][1]

    # print("Q_game_2_2: \n", Q_game_2_2)


end_time = time.time()

print(" \n \n --- Computation time: ", end_time - start_time, " seconds --- \n\n")

# print("Residual history of Q-value: ", res_Q_value_0_0_0)
# print("Residual history of Q-value: ", res_Q_value_2_0_0)

n = list(range(n_iteration))

plt.plot(n, Q_value_0_0_0, label="Q-value[0][0][0]")
plt.plot(n, Q_value_0_0_1, label="Q-value[0][0][1]")
plt.plot(n, Q_value_0_0_2, label="Q-value[0][1][0]")
plt.plot(n, Q_value_0_0_3, label="Q-value[0][1][1]")
plt.plot(n, Q_value_0_0_4, label="Q-value[0][2][0]")
plt.plot(n, Q_value_0_0_5, label="Q-value[0][2][1]")
plt.plot(n, Q_value_0_1_0, label="Q-value[1][0][0]")
plt.plot(n, Q_value_0_1_1, label="Q-value[1][0][1]")
plt.plot(n, Q_value_0_1_2, label="Q-value[1][1][0]")
plt.plot(n, Q_value_0_1_3, label="Q-value[1][1][1]")
plt.plot(n, Q_value_0_1_4, label="Q-value[1][2][0]")
plt.plot(n, Q_value_0_1_5, label="Q-value[1][2][1]")

# plt.plot(n, Q_value_2_0_0, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_1, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_2, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_3, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_4, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_5, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_6, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_7, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_8, label="Q-value[0][0][0]")
# plt.plot(n, Q_value_2_0_9, label="Q-value[0][0][0]")

plt.title("Evolution of Q-values using Nash Q-learning algorithm (pure strategy)")
plt.xlabel("No. of iteration")
plt.ylabel("Q-values")
plt.legend(loc='lower right')
plt.show()
