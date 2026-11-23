"""Auto-generated Nash Q-learning update script."""
import numpy as np
import pygambit as gbt
from SoS_Markov_game_with_transitions import NashEquilibriumMixedStrategy

beta = 0.9
N_ITER = 500

# ---- reward arrays: r_game_<state indices>[a_0][a_1]...[agent]
# agents: 0 = Government, 1 = Manufacturer
# 0_0: Government = low FoM_tech, Manufacturer = manufacture system 1
r_game_0_0 = np.array([[[-3.0, 1.0], [0.0, -4.0], [-2.0, -1.0]], [[-1.0, -2.0], [-1.0, -4.0], [-1.0, 0.0]]])
# 0_1: Government = low FoM_tech, Manufacturer = manufacture system 2
r_game_0_1 = np.array([[[-3.0, 1.0], [0.0, -5.0], [-2.0, -1.0]], [[-1.0, -2.0], [-1.0, -5.0], [-1.0, 0.0]]])
# 0_2: Government = low FoM_tech, Manufacturer = support
r_game_0_2 = np.array([[[0.0, -1.0], [-2.0, -1.0]], [[-1.0, -1.0], [-1.0, 0.0]]])
# 1_0: Government = medium FoM_tech, Manufacturer = manufacture system 1
r_game_1_0 = np.array([[[-3.0, 1.0], [0.0, -4.0], [-2.0, -1.0]], [[-1.0, -2.0], [-1.0, -4.0], [-1.0, 0.0]]])
# 1_1: Government = medium FoM_tech, Manufacturer = manufacture system 2
r_game_1_1 = np.array([[[-3.0, 1.0], [0.0, -5.0], [-2.0, -1.0]], [[-1.0, -2.0], [-1.0, -5.0], [-1.0, 0.0]]])
# 1_2: Government = medium FoM_tech, Manufacturer = support
r_game_1_2 = np.array([[[0.0, -1.0], [-2.0, -1.0]], [[-1.0, -1.0], [-1.0, 0.0]]])
# 2_0: Government = high FoM_tech, Manufacturer = manufacture system 1
r_game_2_0 = np.array([[[-1.0, 1.0], [2.0, -4.0], [0.0, -1.0]], [[-1.0, -2.0], [-1.0, -4.0], [-1.0, 0.0]]])
# 2_1: Government = high FoM_tech, Manufacturer = manufacture system 2
r_game_2_1 = np.array([[[-1.0, 1.0], [2.0, -5.0], [0.0, -1.0]], [[-1.0, -2.0], [-1.0, -5.0], [-1.0, 0.0]]])
# 2_2: Government = high FoM_tech, Manufacturer = support
r_game_2_2 = np.array([[[2.0, -1.0], [0.0, -1.0]], [[-1.0, -1.0], [-1.0, 0.0]]])
# 3_0: Government = operation, Manufacturer = manufacture system 1
r_game_3_0 = np.array([[[10.0, -2.0], [10.0, -4.0], [8.0, -1.0]], [[-1.0, -2.0], [-1.0, -4.0], [-1.0, 0.0]]])
# 3_1: Government = operation, Manufacturer = manufacture system 2
r_game_3_1 = np.array([[[10.0, -2.0], [10.0, -5.0], [8.0, -1.0]], [[-1.0, -2.0], [-1.0, -5.0], [-1.0, 0.0]]])
# 3_2: Government = operation, Manufacturer = support
r_game_3_2 = np.array([[[8.0, 1.0], [8.0, -1.0]], [[-3.0, 1.0], [-1.0, 0.0]]])

# ---- Q tables
Q_game_0_0 = np.zeros_like(r_game_0_0)
Q_game_0_1 = np.zeros_like(r_game_0_1)
Q_game_0_2 = np.zeros_like(r_game_0_2)
Q_game_1_0 = np.zeros_like(r_game_1_0)
Q_game_1_1 = np.zeros_like(r_game_1_1)
Q_game_1_2 = np.zeros_like(r_game_1_2)
Q_game_2_0 = np.zeros_like(r_game_2_0)
Q_game_2_1 = np.zeros_like(r_game_2_1)
Q_game_2_2 = np.zeros_like(r_game_2_2)
Q_game_3_0 = np.zeros_like(r_game_3_0)
Q_game_3_1 = np.zeros_like(r_game_3_1)
Q_game_3_2 = np.zeros_like(r_game_3_2)

for n in range(N_ITER):
    alpha = 1

    # Nash values of every stage game from the current Q tables (each solved once per iteration)
    NashQ_0_0_list = NashEquilibriumMixedStrategy(list(Q_game_0_0))
    NashQ_0_1_list = NashEquilibriumMixedStrategy(list(Q_game_0_1))
    NashQ_0_2_list = NashEquilibriumMixedStrategy(list(Q_game_0_2))
    NashQ_1_0_list = NashEquilibriumMixedStrategy(list(Q_game_1_0))
    NashQ_1_1_list = NashEquilibriumMixedStrategy(list(Q_game_1_1))
    NashQ_1_2_list = NashEquilibriumMixedStrategy(list(Q_game_1_2))
    NashQ_2_0_list = NashEquilibriumMixedStrategy(list(Q_game_2_0))
    NashQ_2_1_list = NashEquilibriumMixedStrategy(list(Q_game_2_1))
    NashQ_2_2_list = NashEquilibriumMixedStrategy(list(Q_game_2_2))
    NashQ_3_0_list = NashEquilibriumMixedStrategy(list(Q_game_3_0))
    NashQ_3_1_list = NashEquilibriumMixedStrategy(list(Q_game_3_1))
    NashQ_3_2_list = NashEquilibriumMixedStrategy(list(Q_game_3_2))

    # --- Game 0_0, 6 action pairs, 2 agents ---
    Q_game_0_0_new = Q_game_0_0.copy()
    ## Action pair (fund, invest) -> next game 1_1
    Q_game_0_0_new[0][0][0] = (1 - alpha) * Q_game_0_0[0][0][0] + alpha * (r_game_0_0[0][0][0] + beta * NashQ_1_1_list[0])   # agent 0: Government
    Q_game_0_0_new[0][0][1] = (1 - alpha) * Q_game_0_0[0][0][1] + alpha * (r_game_0_0[0][0][1] + beta * NashQ_1_1_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, manufacture) -> next game 1_2
    Q_game_0_0_new[0][1][0] = (1 - alpha) * Q_game_0_0[0][1][0] + alpha * (r_game_0_0[0][1][0] + beta * NashQ_1_2_list[0])   # agent 0: Government
    Q_game_0_0_new[0][1][1] = (1 - alpha) * Q_game_0_0[0][1][1] + alpha * (r_game_0_0[0][1][1] + beta * NashQ_1_2_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 1_0
    Q_game_0_0_new[0][2][0] = (1 - alpha) * Q_game_0_0[0][2][0] + alpha * (r_game_0_0[0][2][0] + beta * NashQ_1_0_list[0])   # agent 0: Government
    Q_game_0_0_new[0][2][1] = (1 - alpha) * Q_game_0_0[0][2][1] + alpha * (r_game_0_0[0][2][1] + beta * NashQ_1_0_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 0_1
    Q_game_0_0_new[1][0][0] = (1 - alpha) * Q_game_0_0[1][0][0] + alpha * (r_game_0_0[1][0][0] + beta * NashQ_0_1_list[0])   # agent 0: Government
    Q_game_0_0_new[1][0][1] = (1 - alpha) * Q_game_0_0[1][0][1] + alpha * (r_game_0_0[1][0][1] + beta * NashQ_0_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 0_2
    Q_game_0_0_new[1][1][0] = (1 - alpha) * Q_game_0_0[1][1][0] + alpha * (r_game_0_0[1][1][0] + beta * NashQ_0_2_list[0])   # agent 0: Government
    Q_game_0_0_new[1][1][1] = (1 - alpha) * Q_game_0_0[1][1][1] + alpha * (r_game_0_0[1][1][1] + beta * NashQ_0_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 0_0
    Q_game_0_0_new[1][2][0] = (1 - alpha) * Q_game_0_0[1][2][0] + alpha * (r_game_0_0[1][2][0] + beta * NashQ_0_0_list[0])   # agent 0: Government
    Q_game_0_0_new[1][2][1] = (1 - alpha) * Q_game_0_0[1][2][1] + alpha * (r_game_0_0[1][2][1] + beta * NashQ_0_0_list[1])   # agent 1: Manufacturer
    Q_game_0_0 = Q_game_0_0_new

    # --- Game 0_1, 6 action pairs, 2 agents ---
    Q_game_0_1_new = Q_game_0_1.copy()
    ## Action pair (fund, invest) -> next game 1_1
    Q_game_0_1_new[0][0][0] = (1 - alpha) * Q_game_0_1[0][0][0] + alpha * (r_game_0_1[0][0][0] + beta * NashQ_1_1_list[0])   # agent 0: Government
    Q_game_0_1_new[0][0][1] = (1 - alpha) * Q_game_0_1[0][0][1] + alpha * (r_game_0_1[0][0][1] + beta * NashQ_1_1_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, manufacture) -> next game 1_2
    Q_game_0_1_new[0][1][0] = (1 - alpha) * Q_game_0_1[0][1][0] + alpha * (r_game_0_1[0][1][0] + beta * NashQ_1_2_list[0])   # agent 0: Government
    Q_game_0_1_new[0][1][1] = (1 - alpha) * Q_game_0_1[0][1][1] + alpha * (r_game_0_1[0][1][1] + beta * NashQ_1_2_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 1_1
    Q_game_0_1_new[0][2][0] = (1 - alpha) * Q_game_0_1[0][2][0] + alpha * (r_game_0_1[0][2][0] + beta * NashQ_1_1_list[0])   # agent 0: Government
    Q_game_0_1_new[0][2][1] = (1 - alpha) * Q_game_0_1[0][2][1] + alpha * (r_game_0_1[0][2][1] + beta * NashQ_1_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 0_1
    Q_game_0_1_new[1][0][0] = (1 - alpha) * Q_game_0_1[1][0][0] + alpha * (r_game_0_1[1][0][0] + beta * NashQ_0_1_list[0])   # agent 0: Government
    Q_game_0_1_new[1][0][1] = (1 - alpha) * Q_game_0_1[1][0][1] + alpha * (r_game_0_1[1][0][1] + beta * NashQ_0_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 0_2
    Q_game_0_1_new[1][1][0] = (1 - alpha) * Q_game_0_1[1][1][0] + alpha * (r_game_0_1[1][1][0] + beta * NashQ_0_2_list[0])   # agent 0: Government
    Q_game_0_1_new[1][1][1] = (1 - alpha) * Q_game_0_1[1][1][1] + alpha * (r_game_0_1[1][1][1] + beta * NashQ_0_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 0_1
    Q_game_0_1_new[1][2][0] = (1 - alpha) * Q_game_0_1[1][2][0] + alpha * (r_game_0_1[1][2][0] + beta * NashQ_0_1_list[0])   # agent 0: Government
    Q_game_0_1_new[1][2][1] = (1 - alpha) * Q_game_0_1[1][2][1] + alpha * (r_game_0_1[1][2][1] + beta * NashQ_0_1_list[1])   # agent 1: Manufacturer
    Q_game_0_1 = Q_game_0_1_new

    # --- Game 0_2, 4 action pairs, 2 agents ---
    Q_game_0_2_new = Q_game_0_2.copy()
    ## Action pair (fund, maintain) -> next game 1_2
    Q_game_0_2_new[0][0][0] = (1 - alpha) * Q_game_0_2[0][0][0] + alpha * (r_game_0_2[0][0][0] + beta * NashQ_1_2_list[0])   # agent 0: Government
    Q_game_0_2_new[0][0][1] = (1 - alpha) * Q_game_0_2[0][0][1] + alpha * (r_game_0_2[0][0][1] + beta * NashQ_1_2_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 1_0
    Q_game_0_2_new[0][1][0] = (1 - alpha) * Q_game_0_2[0][1][0] + alpha * (r_game_0_2[0][1][0] + beta * NashQ_1_0_list[0])   # agent 0: Government
    Q_game_0_2_new[0][1][1] = (1 - alpha) * Q_game_0_2[0][1][1] + alpha * (r_game_0_2[0][1][1] + beta * NashQ_1_0_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, maintain) -> next game 0_2
    Q_game_0_2_new[1][0][0] = (1 - alpha) * Q_game_0_2[1][0][0] + alpha * (r_game_0_2[1][0][0] + beta * NashQ_0_2_list[0])   # agent 0: Government
    Q_game_0_2_new[1][0][1] = (1 - alpha) * Q_game_0_2[1][0][1] + alpha * (r_game_0_2[1][0][1] + beta * NashQ_0_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 0_0
    Q_game_0_2_new[1][1][0] = (1 - alpha) * Q_game_0_2[1][1][0] + alpha * (r_game_0_2[1][1][0] + beta * NashQ_0_0_list[0])   # agent 0: Government
    Q_game_0_2_new[1][1][1] = (1 - alpha) * Q_game_0_2[1][1][1] + alpha * (r_game_0_2[1][1][1] + beta * NashQ_0_0_list[1])   # agent 1: Manufacturer
    Q_game_0_2 = Q_game_0_2_new

    # --- Game 1_0, 6 action pairs, 2 agents ---
    Q_game_1_0_new = Q_game_1_0.copy()
    ## Action pair (fund, invest) -> next game 2_1 (p=0.8) + 1_1 (p=0.2)
    Q_game_1_0_new[0][0][0] = (1 - alpha) * Q_game_1_0[0][0][0] + alpha * (r_game_1_0[0][0][0] + beta * (0.8 * NashQ_2_1_list[0] + 0.2 * NashQ_1_1_list[0]))   # agent 0: Government
    Q_game_1_0_new[0][0][1] = (1 - alpha) * Q_game_1_0[0][0][1] + alpha * (r_game_1_0[0][0][1] + beta * (0.8 * NashQ_2_1_list[1] + 0.2 * NashQ_1_1_list[1]))   # agent 1: Manufacturer
    ## Action pair (fund, manufacture) -> next game 2_2 (p=0.8) + 1_2 (p=0.2)
    Q_game_1_0_new[0][1][0] = (1 - alpha) * Q_game_1_0[0][1][0] + alpha * (r_game_1_0[0][1][0] + beta * (0.8 * NashQ_2_2_list[0] + 0.2 * NashQ_1_2_list[0]))   # agent 0: Government
    Q_game_1_0_new[0][1][1] = (1 - alpha) * Q_game_1_0[0][1][1] + alpha * (r_game_1_0[0][1][1] + beta * (0.8 * NashQ_2_2_list[1] + 0.2 * NashQ_1_2_list[1]))   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 2_0 (p=0.8) + 1_0 (p=0.2)
    Q_game_1_0_new[0][2][0] = (1 - alpha) * Q_game_1_0[0][2][0] + alpha * (r_game_1_0[0][2][0] + beta * (0.8 * NashQ_2_0_list[0] + 0.2 * NashQ_1_0_list[0]))   # agent 0: Government
    Q_game_1_0_new[0][2][1] = (1 - alpha) * Q_game_1_0[0][2][1] + alpha * (r_game_1_0[0][2][1] + beta * (0.8 * NashQ_2_0_list[1] + 0.2 * NashQ_1_0_list[1]))   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 1_1
    Q_game_1_0_new[1][0][0] = (1 - alpha) * Q_game_1_0[1][0][0] + alpha * (r_game_1_0[1][0][0] + beta * NashQ_1_1_list[0])   # agent 0: Government
    Q_game_1_0_new[1][0][1] = (1 - alpha) * Q_game_1_0[1][0][1] + alpha * (r_game_1_0[1][0][1] + beta * NashQ_1_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 1_2
    Q_game_1_0_new[1][1][0] = (1 - alpha) * Q_game_1_0[1][1][0] + alpha * (r_game_1_0[1][1][0] + beta * NashQ_1_2_list[0])   # agent 0: Government
    Q_game_1_0_new[1][1][1] = (1 - alpha) * Q_game_1_0[1][1][1] + alpha * (r_game_1_0[1][1][1] + beta * NashQ_1_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 1_0
    Q_game_1_0_new[1][2][0] = (1 - alpha) * Q_game_1_0[1][2][0] + alpha * (r_game_1_0[1][2][0] + beta * NashQ_1_0_list[0])   # agent 0: Government
    Q_game_1_0_new[1][2][1] = (1 - alpha) * Q_game_1_0[1][2][1] + alpha * (r_game_1_0[1][2][1] + beta * NashQ_1_0_list[1])   # agent 1: Manufacturer
    Q_game_1_0 = Q_game_1_0_new

    # --- Game 1_1, 6 action pairs, 2 agents ---
    Q_game_1_1_new = Q_game_1_1.copy()
    ## Action pair (fund, invest) -> next game 2_1 (p=0.8) + 1_1 (p=0.2)
    Q_game_1_1_new[0][0][0] = (1 - alpha) * Q_game_1_1[0][0][0] + alpha * (r_game_1_1[0][0][0] + beta * (0.8 * NashQ_2_1_list[0] + 0.2 * NashQ_1_1_list[0]))   # agent 0: Government
    Q_game_1_1_new[0][0][1] = (1 - alpha) * Q_game_1_1[0][0][1] + alpha * (r_game_1_1[0][0][1] + beta * (0.8 * NashQ_2_1_list[1] + 0.2 * NashQ_1_1_list[1]))   # agent 1: Manufacturer
    ## Action pair (fund, manufacture) -> next game 2_2 (p=0.8) + 1_2 (p=0.2)
    Q_game_1_1_new[0][1][0] = (1 - alpha) * Q_game_1_1[0][1][0] + alpha * (r_game_1_1[0][1][0] + beta * (0.8 * NashQ_2_2_list[0] + 0.2 * NashQ_1_2_list[0]))   # agent 0: Government
    Q_game_1_1_new[0][1][1] = (1 - alpha) * Q_game_1_1[0][1][1] + alpha * (r_game_1_1[0][1][1] + beta * (0.8 * NashQ_2_2_list[1] + 0.2 * NashQ_1_2_list[1]))   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 2_1 (p=0.8) + 1_1 (p=0.2)
    Q_game_1_1_new[0][2][0] = (1 - alpha) * Q_game_1_1[0][2][0] + alpha * (r_game_1_1[0][2][0] + beta * (0.8 * NashQ_2_1_list[0] + 0.2 * NashQ_1_1_list[0]))   # agent 0: Government
    Q_game_1_1_new[0][2][1] = (1 - alpha) * Q_game_1_1[0][2][1] + alpha * (r_game_1_1[0][2][1] + beta * (0.8 * NashQ_2_1_list[1] + 0.2 * NashQ_1_1_list[1]))   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 1_1
    Q_game_1_1_new[1][0][0] = (1 - alpha) * Q_game_1_1[1][0][0] + alpha * (r_game_1_1[1][0][0] + beta * NashQ_1_1_list[0])   # agent 0: Government
    Q_game_1_1_new[1][0][1] = (1 - alpha) * Q_game_1_1[1][0][1] + alpha * (r_game_1_1[1][0][1] + beta * NashQ_1_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 1_2
    Q_game_1_1_new[1][1][0] = (1 - alpha) * Q_game_1_1[1][1][0] + alpha * (r_game_1_1[1][1][0] + beta * NashQ_1_2_list[0])   # agent 0: Government
    Q_game_1_1_new[1][1][1] = (1 - alpha) * Q_game_1_1[1][1][1] + alpha * (r_game_1_1[1][1][1] + beta * NashQ_1_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 1_1
    Q_game_1_1_new[1][2][0] = (1 - alpha) * Q_game_1_1[1][2][0] + alpha * (r_game_1_1[1][2][0] + beta * NashQ_1_1_list[0])   # agent 0: Government
    Q_game_1_1_new[1][2][1] = (1 - alpha) * Q_game_1_1[1][2][1] + alpha * (r_game_1_1[1][2][1] + beta * NashQ_1_1_list[1])   # agent 1: Manufacturer
    Q_game_1_1 = Q_game_1_1_new

    # --- Game 1_2, 4 action pairs, 2 agents ---
    Q_game_1_2_new = Q_game_1_2.copy()
    ## Action pair (fund, maintain) -> next game 2_2 (p=0.8) + 1_2 (p=0.2)
    Q_game_1_2_new[0][0][0] = (1 - alpha) * Q_game_1_2[0][0][0] + alpha * (r_game_1_2[0][0][0] + beta * (0.8 * NashQ_2_2_list[0] + 0.2 * NashQ_1_2_list[0]))   # agent 0: Government
    Q_game_1_2_new[0][0][1] = (1 - alpha) * Q_game_1_2[0][0][1] + alpha * (r_game_1_2[0][0][1] + beta * (0.8 * NashQ_2_2_list[1] + 0.2 * NashQ_1_2_list[1]))   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 2_0 (p=0.8) + 1_0 (p=0.2)
    Q_game_1_2_new[0][1][0] = (1 - alpha) * Q_game_1_2[0][1][0] + alpha * (r_game_1_2[0][1][0] + beta * (0.8 * NashQ_2_0_list[0] + 0.2 * NashQ_1_0_list[0]))   # agent 0: Government
    Q_game_1_2_new[0][1][1] = (1 - alpha) * Q_game_1_2[0][1][1] + alpha * (r_game_1_2[0][1][1] + beta * (0.8 * NashQ_2_0_list[1] + 0.2 * NashQ_1_0_list[1]))   # agent 1: Manufacturer
    ## Action pair (wait, maintain) -> next game 1_2
    Q_game_1_2_new[1][0][0] = (1 - alpha) * Q_game_1_2[1][0][0] + alpha * (r_game_1_2[1][0][0] + beta * NashQ_1_2_list[0])   # agent 0: Government
    Q_game_1_2_new[1][0][1] = (1 - alpha) * Q_game_1_2[1][0][1] + alpha * (r_game_1_2[1][0][1] + beta * NashQ_1_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 1_0
    Q_game_1_2_new[1][1][0] = (1 - alpha) * Q_game_1_2[1][1][0] + alpha * (r_game_1_2[1][1][0] + beta * NashQ_1_0_list[0])   # agent 0: Government
    Q_game_1_2_new[1][1][1] = (1 - alpha) * Q_game_1_2[1][1][1] + alpha * (r_game_1_2[1][1][1] + beta * NashQ_1_0_list[1])   # agent 1: Manufacturer
    Q_game_1_2 = Q_game_1_2_new

    # --- Game 2_0, 6 action pairs, 2 agents ---
    Q_game_2_0_new = Q_game_2_0.copy()
    ## Action pair (fund, invest) -> next game 3_1
    Q_game_2_0_new[0][0][0] = (1 - alpha) * Q_game_2_0[0][0][0] + alpha * (r_game_2_0[0][0][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_2_0_new[0][0][1] = (1 - alpha) * Q_game_2_0[0][0][1] + alpha * (r_game_2_0[0][0][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, manufacture) -> next game 3_2
    Q_game_2_0_new[0][1][0] = (1 - alpha) * Q_game_2_0[0][1][0] + alpha * (r_game_2_0[0][1][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_2_0_new[0][1][1] = (1 - alpha) * Q_game_2_0[0][1][1] + alpha * (r_game_2_0[0][1][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 3_0
    Q_game_2_0_new[0][2][0] = (1 - alpha) * Q_game_2_0[0][2][0] + alpha * (r_game_2_0[0][2][0] + beta * NashQ_3_0_list[0])   # agent 0: Government
    Q_game_2_0_new[0][2][1] = (1 - alpha) * Q_game_2_0[0][2][1] + alpha * (r_game_2_0[0][2][1] + beta * NashQ_3_0_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 2_1
    Q_game_2_0_new[1][0][0] = (1 - alpha) * Q_game_2_0[1][0][0] + alpha * (r_game_2_0[1][0][0] + beta * NashQ_2_1_list[0])   # agent 0: Government
    Q_game_2_0_new[1][0][1] = (1 - alpha) * Q_game_2_0[1][0][1] + alpha * (r_game_2_0[1][0][1] + beta * NashQ_2_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 2_2
    Q_game_2_0_new[1][1][0] = (1 - alpha) * Q_game_2_0[1][1][0] + alpha * (r_game_2_0[1][1][0] + beta * NashQ_2_2_list[0])   # agent 0: Government
    Q_game_2_0_new[1][1][1] = (1 - alpha) * Q_game_2_0[1][1][1] + alpha * (r_game_2_0[1][1][1] + beta * NashQ_2_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 2_0
    Q_game_2_0_new[1][2][0] = (1 - alpha) * Q_game_2_0[1][2][0] + alpha * (r_game_2_0[1][2][0] + beta * NashQ_2_0_list[0])   # agent 0: Government
    Q_game_2_0_new[1][2][1] = (1 - alpha) * Q_game_2_0[1][2][1] + alpha * (r_game_2_0[1][2][1] + beta * NashQ_2_0_list[1])   # agent 1: Manufacturer
    Q_game_2_0 = Q_game_2_0_new

    # --- Game 2_1, 6 action pairs, 2 agents ---
    Q_game_2_1_new = Q_game_2_1.copy()
    ## Action pair (fund, invest) -> next game 3_1
    Q_game_2_1_new[0][0][0] = (1 - alpha) * Q_game_2_1[0][0][0] + alpha * (r_game_2_1[0][0][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_2_1_new[0][0][1] = (1 - alpha) * Q_game_2_1[0][0][1] + alpha * (r_game_2_1[0][0][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, manufacture) -> next game 3_2
    Q_game_2_1_new[0][1][0] = (1 - alpha) * Q_game_2_1[0][1][0] + alpha * (r_game_2_1[0][1][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_2_1_new[0][1][1] = (1 - alpha) * Q_game_2_1[0][1][1] + alpha * (r_game_2_1[0][1][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 3_1
    Q_game_2_1_new[0][2][0] = (1 - alpha) * Q_game_2_1[0][2][0] + alpha * (r_game_2_1[0][2][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_2_1_new[0][2][1] = (1 - alpha) * Q_game_2_1[0][2][1] + alpha * (r_game_2_1[0][2][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 2_1
    Q_game_2_1_new[1][0][0] = (1 - alpha) * Q_game_2_1[1][0][0] + alpha * (r_game_2_1[1][0][0] + beta * NashQ_2_1_list[0])   # agent 0: Government
    Q_game_2_1_new[1][0][1] = (1 - alpha) * Q_game_2_1[1][0][1] + alpha * (r_game_2_1[1][0][1] + beta * NashQ_2_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 2_2
    Q_game_2_1_new[1][1][0] = (1 - alpha) * Q_game_2_1[1][1][0] + alpha * (r_game_2_1[1][1][0] + beta * NashQ_2_2_list[0])   # agent 0: Government
    Q_game_2_1_new[1][1][1] = (1 - alpha) * Q_game_2_1[1][1][1] + alpha * (r_game_2_1[1][1][1] + beta * NashQ_2_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 2_1
    Q_game_2_1_new[1][2][0] = (1 - alpha) * Q_game_2_1[1][2][0] + alpha * (r_game_2_1[1][2][0] + beta * NashQ_2_1_list[0])   # agent 0: Government
    Q_game_2_1_new[1][2][1] = (1 - alpha) * Q_game_2_1[1][2][1] + alpha * (r_game_2_1[1][2][1] + beta * NashQ_2_1_list[1])   # agent 1: Manufacturer
    Q_game_2_1 = Q_game_2_1_new

    # --- Game 2_2, 4 action pairs, 2 agents ---
    Q_game_2_2_new = Q_game_2_2.copy()
    ## Action pair (fund, maintain) -> next game 3_2
    Q_game_2_2_new[0][0][0] = (1 - alpha) * Q_game_2_2[0][0][0] + alpha * (r_game_2_2[0][0][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_2_2_new[0][0][1] = (1 - alpha) * Q_game_2_2[0][0][1] + alpha * (r_game_2_2[0][0][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (fund, idle) -> next game 3_0
    Q_game_2_2_new[0][1][0] = (1 - alpha) * Q_game_2_2[0][1][0] + alpha * (r_game_2_2[0][1][0] + beta * NashQ_3_0_list[0])   # agent 0: Government
    Q_game_2_2_new[0][1][1] = (1 - alpha) * Q_game_2_2[0][1][1] + alpha * (r_game_2_2[0][1][1] + beta * NashQ_3_0_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, maintain) -> next game 2_2
    Q_game_2_2_new[1][0][0] = (1 - alpha) * Q_game_2_2[1][0][0] + alpha * (r_game_2_2[1][0][0] + beta * NashQ_2_2_list[0])   # agent 0: Government
    Q_game_2_2_new[1][0][1] = (1 - alpha) * Q_game_2_2[1][0][1] + alpha * (r_game_2_2[1][0][1] + beta * NashQ_2_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 2_0
    Q_game_2_2_new[1][1][0] = (1 - alpha) * Q_game_2_2[1][1][0] + alpha * (r_game_2_2[1][1][0] + beta * NashQ_2_0_list[0])   # agent 0: Government
    Q_game_2_2_new[1][1][1] = (1 - alpha) * Q_game_2_2[1][1][1] + alpha * (r_game_2_2[1][1][1] + beta * NashQ_2_0_list[1])   # agent 1: Manufacturer
    Q_game_2_2 = Q_game_2_2_new

    # --- Game 3_0, 6 action pairs, 2 agents ---
    Q_game_3_0_new = Q_game_3_0.copy()
    ## Action pair (acquire, invest) -> next game 3_1
    Q_game_3_0_new[0][0][0] = (1 - alpha) * Q_game_3_0[0][0][0] + alpha * (r_game_3_0[0][0][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_3_0_new[0][0][1] = (1 - alpha) * Q_game_3_0[0][0][1] + alpha * (r_game_3_0[0][0][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (acquire, manufacture) -> next game 3_2
    Q_game_3_0_new[0][1][0] = (1 - alpha) * Q_game_3_0[0][1][0] + alpha * (r_game_3_0[0][1][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_3_0_new[0][1][1] = (1 - alpha) * Q_game_3_0[0][1][1] + alpha * (r_game_3_0[0][1][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (acquire, idle) -> next game 3_0
    Q_game_3_0_new[0][2][0] = (1 - alpha) * Q_game_3_0[0][2][0] + alpha * (r_game_3_0[0][2][0] + beta * NashQ_3_0_list[0])   # agent 0: Government
    Q_game_3_0_new[0][2][1] = (1 - alpha) * Q_game_3_0[0][2][1] + alpha * (r_game_3_0[0][2][1] + beta * NashQ_3_0_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 3_1
    Q_game_3_0_new[1][0][0] = (1 - alpha) * Q_game_3_0[1][0][0] + alpha * (r_game_3_0[1][0][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_3_0_new[1][0][1] = (1 - alpha) * Q_game_3_0[1][0][1] + alpha * (r_game_3_0[1][0][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 3_2
    Q_game_3_0_new[1][1][0] = (1 - alpha) * Q_game_3_0[1][1][0] + alpha * (r_game_3_0[1][1][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_3_0_new[1][1][1] = (1 - alpha) * Q_game_3_0[1][1][1] + alpha * (r_game_3_0[1][1][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 3_0
    Q_game_3_0_new[1][2][0] = (1 - alpha) * Q_game_3_0[1][2][0] + alpha * (r_game_3_0[1][2][0] + beta * NashQ_3_0_list[0])   # agent 0: Government
    Q_game_3_0_new[1][2][1] = (1 - alpha) * Q_game_3_0[1][2][1] + alpha * (r_game_3_0[1][2][1] + beta * NashQ_3_0_list[1])   # agent 1: Manufacturer
    Q_game_3_0 = Q_game_3_0_new

    # --- Game 3_1, 6 action pairs, 2 agents ---
    Q_game_3_1_new = Q_game_3_1.copy()
    ## Action pair (acquire, invest) -> next game 3_1
    Q_game_3_1_new[0][0][0] = (1 - alpha) * Q_game_3_1[0][0][0] + alpha * (r_game_3_1[0][0][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_3_1_new[0][0][1] = (1 - alpha) * Q_game_3_1[0][0][1] + alpha * (r_game_3_1[0][0][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (acquire, manufacture) -> next game 3_2
    Q_game_3_1_new[0][1][0] = (1 - alpha) * Q_game_3_1[0][1][0] + alpha * (r_game_3_1[0][1][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_3_1_new[0][1][1] = (1 - alpha) * Q_game_3_1[0][1][1] + alpha * (r_game_3_1[0][1][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (acquire, idle) -> next game 3_1
    Q_game_3_1_new[0][2][0] = (1 - alpha) * Q_game_3_1[0][2][0] + alpha * (r_game_3_1[0][2][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_3_1_new[0][2][1] = (1 - alpha) * Q_game_3_1[0][2][1] + alpha * (r_game_3_1[0][2][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, invest) -> next game 3_1
    Q_game_3_1_new[1][0][0] = (1 - alpha) * Q_game_3_1[1][0][0] + alpha * (r_game_3_1[1][0][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_3_1_new[1][0][1] = (1 - alpha) * Q_game_3_1[1][0][1] + alpha * (r_game_3_1[1][0][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, manufacture) -> next game 3_2
    Q_game_3_1_new[1][1][0] = (1 - alpha) * Q_game_3_1[1][1][0] + alpha * (r_game_3_1[1][1][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_3_1_new[1][1][1] = (1 - alpha) * Q_game_3_1[1][1][1] + alpha * (r_game_3_1[1][1][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 3_1
    Q_game_3_1_new[1][2][0] = (1 - alpha) * Q_game_3_1[1][2][0] + alpha * (r_game_3_1[1][2][0] + beta * NashQ_3_1_list[0])   # agent 0: Government
    Q_game_3_1_new[1][2][1] = (1 - alpha) * Q_game_3_1[1][2][1] + alpha * (r_game_3_1[1][2][1] + beta * NashQ_3_1_list[1])   # agent 1: Manufacturer
    Q_game_3_1 = Q_game_3_1_new

    # --- Game 3_2, 4 action pairs, 2 agents ---
    Q_game_3_2_new = Q_game_3_2.copy()
    ## Action pair (acquire, maintain) -> next game 3_2
    Q_game_3_2_new[0][0][0] = (1 - alpha) * Q_game_3_2[0][0][0] + alpha * (r_game_3_2[0][0][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_3_2_new[0][0][1] = (1 - alpha) * Q_game_3_2[0][0][1] + alpha * (r_game_3_2[0][0][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (acquire, idle) -> next game 3_0
    Q_game_3_2_new[0][1][0] = (1 - alpha) * Q_game_3_2[0][1][0] + alpha * (r_game_3_2[0][1][0] + beta * NashQ_3_0_list[0])   # agent 0: Government
    Q_game_3_2_new[0][1][1] = (1 - alpha) * Q_game_3_2[0][1][1] + alpha * (r_game_3_2[0][1][1] + beta * NashQ_3_0_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, maintain) -> next game 3_2
    Q_game_3_2_new[1][0][0] = (1 - alpha) * Q_game_3_2[1][0][0] + alpha * (r_game_3_2[1][0][0] + beta * NashQ_3_2_list[0])   # agent 0: Government
    Q_game_3_2_new[1][0][1] = (1 - alpha) * Q_game_3_2[1][0][1] + alpha * (r_game_3_2[1][0][1] + beta * NashQ_3_2_list[1])   # agent 1: Manufacturer
    ## Action pair (wait, idle) -> next game 3_0
    Q_game_3_2_new[1][1][0] = (1 - alpha) * Q_game_3_2[1][1][0] + alpha * (r_game_3_2[1][1][0] + beta * NashQ_3_0_list[0])   # agent 0: Government
    Q_game_3_2_new[1][1][1] = (1 - alpha) * Q_game_3_2[1][1][1] + alpha * (r_game_3_2[1][1][1] + beta * NashQ_3_0_list[1])   # agent 1: Manufacturer
    Q_game_3_2 = Q_game_3_2_new


# =====================================================================================
# Best policy from the learned Q-values
#   1. solve the Nash equilibrium of every stage game on its Q array -> strategy per agent
#   2. start in a stage game, play the equilibrium joint action, move to the next game
#      through the transitions, repeat
# =====================================================================================
AGENTS = ["Government", "Manufacturer"]

STATES = {   # stage game -> state of each agent
    '0_0': ('low FoM_tech', 'manufacture system 1'),
    '0_1': ('low FoM_tech', 'manufacture system 2'),
    '0_2': ('low FoM_tech', 'support'),
    '1_0': ('medium FoM_tech', 'manufacture system 1'),
    '1_1': ('medium FoM_tech', 'manufacture system 2'),
    '1_2': ('medium FoM_tech', 'support'),
    '2_0': ('high FoM_tech', 'manufacture system 1'),
    '2_1': ('high FoM_tech', 'manufacture system 2'),
    '2_2': ('high FoM_tech', 'support'),
    '3_0': ('operation', 'manufacture system 1'),
    '3_1': ('operation', 'manufacture system 2'),
    '3_2': ('operation', 'support'),
}
ACTIONS = {  # stage game -> available actions of each agent, in Q-array index order
    '0_0': [['fund', 'wait'], ['invest', 'manufacture', 'idle']],
    '0_1': [['fund', 'wait'], ['invest', 'manufacture', 'idle']],
    '0_2': [['fund', 'wait'], ['maintain', 'idle']],
    '1_0': [['fund', 'wait'], ['invest', 'manufacture', 'idle']],
    '1_1': [['fund', 'wait'], ['invest', 'manufacture', 'idle']],
    '1_2': [['fund', 'wait'], ['maintain', 'idle']],
    '2_0': [['fund', 'wait'], ['invest', 'manufacture', 'idle']],
    '2_1': [['fund', 'wait'], ['invest', 'manufacture', 'idle']],
    '2_2': [['fund', 'wait'], ['maintain', 'idle']],
    '3_0': [['acquire', 'wait'], ['invest', 'manufacture', 'idle']],
    '3_1': [['acquire', 'wait'], ['invest', 'manufacture', 'idle']],
    '3_2': [['acquire', 'wait'], ['maintain', 'idle']],
}
NEXT_GAME = {  # stage game -> {joint action indices: [(next stage game, probability), ...]}
    '0_0': {
        (0, 0): [('1_1', 1.0)],   # (fund, invest)
        (0, 1): [('1_2', 1.0)],   # (fund, manufacture)
        (0, 2): [('1_0', 1.0)],   # (fund, idle)
        (1, 0): [('0_1', 1.0)],   # (wait, invest)
        (1, 1): [('0_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('0_0', 1.0)],   # (wait, idle)
    },
    '0_1': {
        (0, 0): [('1_1', 1.0)],   # (fund, invest)
        (0, 1): [('1_2', 1.0)],   # (fund, manufacture)
        (0, 2): [('1_1', 1.0)],   # (fund, idle)
        (1, 0): [('0_1', 1.0)],   # (wait, invest)
        (1, 1): [('0_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('0_1', 1.0)],   # (wait, idle)
    },
    '0_2': {
        (0, 0): [('1_2', 1.0)],   # (fund, maintain)
        (0, 1): [('1_0', 1.0)],   # (fund, idle)
        (1, 0): [('0_2', 1.0)],   # (wait, maintain)
        (1, 1): [('0_0', 1.0)],   # (wait, idle)
    },
    '1_0': {
        (0, 0): [('2_1', 0.8), ('1_1', 0.2)],   # (fund, invest)
        (0, 1): [('2_2', 0.8), ('1_2', 0.2)],   # (fund, manufacture)
        (0, 2): [('2_0', 0.8), ('1_0', 0.2)],   # (fund, idle)
        (1, 0): [('1_1', 1.0)],   # (wait, invest)
        (1, 1): [('1_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('1_0', 1.0)],   # (wait, idle)
    },
    '1_1': {
        (0, 0): [('2_1', 0.8), ('1_1', 0.2)],   # (fund, invest)
        (0, 1): [('2_2', 0.8), ('1_2', 0.2)],   # (fund, manufacture)
        (0, 2): [('2_1', 0.8), ('1_1', 0.2)],   # (fund, idle)
        (1, 0): [('1_1', 1.0)],   # (wait, invest)
        (1, 1): [('1_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('1_1', 1.0)],   # (wait, idle)
    },
    '1_2': {
        (0, 0): [('2_2', 0.8), ('1_2', 0.2)],   # (fund, maintain)
        (0, 1): [('2_0', 0.8), ('1_0', 0.2)],   # (fund, idle)
        (1, 0): [('1_2', 1.0)],   # (wait, maintain)
        (1, 1): [('1_0', 1.0)],   # (wait, idle)
    },
    '2_0': {
        (0, 0): [('3_1', 1.0)],   # (fund, invest)
        (0, 1): [('3_2', 1.0)],   # (fund, manufacture)
        (0, 2): [('3_0', 1.0)],   # (fund, idle)
        (1, 0): [('2_1', 1.0)],   # (wait, invest)
        (1, 1): [('2_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('2_0', 1.0)],   # (wait, idle)
    },
    '2_1': {
        (0, 0): [('3_1', 1.0)],   # (fund, invest)
        (0, 1): [('3_2', 1.0)],   # (fund, manufacture)
        (0, 2): [('3_1', 1.0)],   # (fund, idle)
        (1, 0): [('2_1', 1.0)],   # (wait, invest)
        (1, 1): [('2_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('2_1', 1.0)],   # (wait, idle)
    },
    '2_2': {
        (0, 0): [('3_2', 1.0)],   # (fund, maintain)
        (0, 1): [('3_0', 1.0)],   # (fund, idle)
        (1, 0): [('2_2', 1.0)],   # (wait, maintain)
        (1, 1): [('2_0', 1.0)],   # (wait, idle)
    },
    '3_0': {
        (0, 0): [('3_1', 1.0)],   # (acquire, invest)
        (0, 1): [('3_2', 1.0)],   # (acquire, manufacture)
        (0, 2): [('3_0', 1.0)],   # (acquire, idle)
        (1, 0): [('3_1', 1.0)],   # (wait, invest)
        (1, 1): [('3_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('3_0', 1.0)],   # (wait, idle)
    },
    '3_1': {
        (0, 0): [('3_1', 1.0)],   # (acquire, invest)
        (0, 1): [('3_2', 1.0)],   # (acquire, manufacture)
        (0, 2): [('3_1', 1.0)],   # (acquire, idle)
        (1, 0): [('3_1', 1.0)],   # (wait, invest)
        (1, 1): [('3_2', 1.0)],   # (wait, manufacture)
        (1, 2): [('3_1', 1.0)],   # (wait, idle)
    },
    '3_2': {
        (0, 0): [('3_2', 1.0)],   # (acquire, maintain)
        (0, 1): [('3_0', 1.0)],   # (acquire, idle)
        (1, 0): [('3_2', 1.0)],   # (wait, maintain)
        (1, 1): [('3_0', 1.0)],   # (wait, idle)
    },
}

Q_GAMES = {g: globals()[f"Q_game_{g}"] for g in STATES}    # learned Q-values per stage game
R_GAMES = {g: globals()[f"r_game_{g}"] for g in STATES}    # immediate rewards per stage game


def NashEquilibriumStrategy(Q_game):
    """Like NashEquilibriumMixedStrategy, but also returns the strategies.
    Returns (probabilities per agent, Nash value per agent)."""
    Q = np.asarray(Q_game, dtype=float)
    g = gbt.Game.from_arrays(*[Q[..., i] for i in range(Q.shape[-1])])
    eq = gbt.nash.logit_solve(g).equilibria[0]
    probabilities = [np.array([float(eq[s]) for s in p.strategies]) for p in g.players]
    values = [float(eq.payoff(p)) for p in g.players]
    return probabilities, values


def best_policy(Q_games):
    """Equilibrium strategy and Nash value of every stage game."""
    policy = {}
    for name, Q in Q_games.items():
        probs, values = NashEquilibriumStrategy(Q)
        policy[name] = {"probabilities": probs, "values": values}
    return policy


def follow_policy(policy, start_game, n_steps, action_mode="most_likely",
                  transition_mode="most_likely", seed=None):
    """Sequence of stage games and joint actions when every agent follows the policy.
    action_mode     : "most_likely" -> each agent plays its highest-probability action
                      "sample"      -> draw the action from the (mixed) equilibrium strategy
    transition_mode : "most_likely" -> go to the most probable next game
                      "sample"      -> draw the next game from the transition probabilities"""
    rng = np.random.default_rng(seed)
    game, steps, total = start_game, [], np.zeros(len(AGENTS))
    for t in range(n_steps):
        probs = policy[game]["probabilities"]
        if action_mode == "most_likely":
            idx = tuple(int(np.argmax(p)) for p in probs)
        else:
            idx = tuple(int(rng.choice(len(p), p=np.clip(p, 0, None) / np.clip(p, 0, None).sum()))
                        for p in probs)
        outcomes = NEXT_GAME[game][idx]
        if transition_mode == "most_likely":
            nxt = max(outcomes, key=lambda o: o[1])[0]
        else:
            nxt = outcomes[int(rng.choice(len(outcomes), p=[p for _, p in outcomes]))][0]
        reward = R_GAMES[game][idx]
        total = total + reward
        steps.append({
            "t": t,
            "game": game,
            "states": STATES[game],
            "joint_action": tuple(ACTIONS[game][k][j] for k, j in enumerate(idx)),
            "action_probability": tuple(float(probs[k][j]) for k, j in enumerate(idx)),
            "reward": reward.copy(),
            "accumulated_reward": total.copy(),
            "next_game": nxt,
        })
        game = nxt
    return steps


def print_sequence(steps):
    for s in steps:
        acts = ", ".join(s["joint_action"])
        prob = ", ".join(f"{p:.2f}" for p in s["action_probability"])
        print(f"t={s['t']:<3} game {s['game']}  [{' | '.join(s['states'])}]")
        print(f"       actions ({acts})  prob ({prob})  r = {s['reward'].round(3).tolist()}  "
              f"cum = {s['accumulated_reward'].round(3).tolist()}  -> next game {s['next_game']}")


def policy_monte_carlo(policy, start_game, n_steps, n_runs=1000, seed=None):
    """Many runs with sampled actions and transitions: frequency of each game sequence
    and mean accumulated reward."""
    rng = np.random.default_rng(seed)
    counts, totals = {}, []
    for _ in range(n_runs):
        steps = follow_policy(policy, start_game, n_steps, "sample", "sample",
                              seed=int(rng.integers(2**31)))
        seq = tuple(s["game"] for s in steps) + (steps[-1]["next_game"],)
        counts[seq] = counts.get(seq, 0) + 1
        totals.append(steps[-1]["accumulated_reward"])
    return sorted(counts.items(), key=lambda kv: -kv[1]), np.array(totals)


if __name__ == '__main__':
    START_GAME = "0_0"     # Government = low FoM_tech, Manufacturer = manufacture system 1
    N_STEPS = 7

    policy = best_policy(Q_GAMES)

    print("Equilibrium policy per stage game:")
    for g, pol in policy.items():
        mix = " | ".join(
            f"{AGENTS[k]}: " + ", ".join(f"{a} {p:.2f}" for a, p in zip(ACTIONS[g][k], pol["probabilities"][k]) if p > 0.01)
            for k in range(len(AGENTS)))
        print(f"  {g} {STATES[g]}: {mix}   Nash value = {np.round(pol['values'], 3).tolist()}")

    print("\nBest-policy sequence (most likely actions and transitions):")
    print_sequence(follow_policy(policy, START_GAME, N_STEPS))

    print("\nWith stochastic transitions (1000 sampled runs):")
    sequences, totals = policy_monte_carlo(policy, START_GAME, N_STEPS, n_runs=1000, seed=1)
    for seq, count in sequences[:3]:
        print(f"  {count:>4} runs: " + " -> ".join(seq))
    print("  mean accumulated reward:", {a: round(float(v), 3) for a, v in zip(AGENTS, totals.mean(axis=0))})