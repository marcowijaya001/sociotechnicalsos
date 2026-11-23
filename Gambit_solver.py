import numpy as np
import pygambit as gbt
from Game_model import *

# Same format as nash_nplayer.py: Q[a0, a1, a2] = [q0, q1, q2]
# 3-player matching pennies: no pure Nash equilibrium

# Format 1: a unified array
# Q = np.array([
#     [   # a0 = 0
#         [[ 1,  1, -1], [ 1, -1,  1]],   # a1 = 0 :  a2 = 0, a2 = 1
#         [[-1, -1, -1], [-1,  1,  1]],   # a1 = 1 :  a2 = 0, a2 = 1
#     ],
#     [   # a0 = 1
#         [[-1,  1,  1], [-1, -1, -1]],   # a1 = 0 :  a2 = 0, a2 = 1
#         [[ 1, -1,  1], [ 1,  1, -1]],   # a1 = 1 :  a2 = 0, a2 = 1
#     ],
# ])

# Q = np.array([
#         # a2 = 0: rows = a0, columns = a1
#         [[[10, 3, 7], [0, 4, 4]],
#          [[3, 2, 4], [2, 9, 2]]],
#         # a2 = 1
#         [[[0, 1, 2], [3, 3, 1]],
#          [[5, 0, 1], [2, 5, 3]]],
#     ])

# Q = np.array([
#         [  # a0 = 0
#             [[3, 3, 3], [0, 4, 7]], # a1 = 0 : a2 = 0, a2 = 1
#             [[2, 4, 2], [1, 0, 3]],
#             [[0, 0, 4], [1, 1, 1]]],   # a1 = 1
#         [  # a0 = 1
#             [[4, 0, 0], [1, 1, 1]],
#             [[4, 1, 1], [2, 2, 2]],
#             [[1, 1, 1], [2, 3, 2]]],
#     ])

Q = np.array([
            [[0,1], [1,0], [1,0]],
            [[1,0], [0,1], [1,0]],
            [[1,0], [1,0], [0,1]],
    ])

# Format 2: split for each joint actions
# Q = np.zeros((2, 2, 2, 3))
# #   a0 a1 a2      q0  q1  q2
# Q[0, 0, 0] = [ 1,  1, -1]
# Q[0, 0, 1] = [ 1, -1,  1]
# Q[0, 1, 0] = [-1, -1, -1]
# Q[0, 1, 1] = [-1,  1,  1]
# Q[1, 0, 0] = [-1,  1,  1]
# Q[1, 0, 1] = [-1, -1, -1]
# Q[1, 1, 0] = [ 1, -1,  1]
# Q[1, 1, 1] = [ 1,  1, -1]

# Format 3: split per agent
# Q0 = np.array([[[ 1,  1], [-1, -1]],
#                [[-1, -1], [ 1,  1]]])
# Q1 = np.array([[[ 1, -1], [-1,  1]],
#                [[ 1, -1], [-1,  1]]])
# Q2 = np.array([[[-1,  1], [-1,  1]],
#                [[ 1, -1], [ 1, -1]]])
# Q = np.stack([Q0, Q1, Q2], axis=-1)

# Q = np.asarray(Q_game_2_0)

# g = gbt.Game.from_arrays(*[Q[..., i] for i in range(Q.shape[-1])])
# # Solver for mixed strategy for finding Nash equilibriium points
# result = gbt.nash.logit_solve(g)
# # result = gbt.nash.enumpure_solve(g)
# for eq in result.equilibria:
#     print([[float(eq[s]) for s in p.strategies] for p in g.players])  # probabilities
#     print([float(eq.payoff(p)) for p in g.players])                    # expected values
#     print(float(eq.max_regret()))                                      # ~0 means it is a Nash equilibrium

# Solver for pure strategy for finding Nash equilibriium points
# result = gbt.nash.enumpure_solve(g)
# msp = result.equilibria[0]
# print(msp)
# for player in g.players:
#     print(f"{player.label} plays the equilibrium strategy:")
#     print(f"Payoff: {msp.payoff(player.label)}")
#     print()

def NashEquilibriumMixedStartegy(Q_i):
    # Converting format of array
    Q = np.asarray(Q_i)
    g = gbt.Game.from_arrays(*[Q[..., i] for i in range(Q.shape[-1])])
    # Solver for mixed strategy for finding Nash equilibriium points
    result = gbt.nash.logit_solve(g)
    for eq in result.equilibria:
        probabilities = [[float(eq[s]) for s in p.strategies] for p in g.players]  # probabilities
        Nash_values = [float(eq.payoff(p)) for p in g.players]  # expected values
        max_regret = float(eq.max_regret())

    return Nash_values, probabilities, max_regret

def NashEquilibriumPureStartegy(Q_i):
    Q = np.asarray(Q_i)
    g = gbt.Game.from_arrays(*[Q[..., i] for i in range(Q.shape[-1])])
    result = gbt.nash.enumpure_solve(g)

    Nash_values_list, Nash_points_list, probabilities_list, max_regret_list = [], [], [], []
    for eq in result.equilibria:
        probabilities = [[float(eq[s]) for s in p.strategies] for p in g.players]
        Nash_point = [int(np.argmax(p)) for p in probabilities]   # [a0, a1, a2]
        Nash_values_list.append([float(eq.payoff(p)) for p in g.players])
        Nash_points_list.append(Nash_point)
        probabilities_list.append(probabilities)
        max_regret_list.append(float(eq.max_regret()))

    return Nash_values_list, Nash_points_list, probabilities_list, max_regret_list


Q_game_0_0_0 = [[[[-3.677536300959667, -3.1195038327175744, -0.5104001962473402], [-3.677536300959752, -3.1195038340303256, 0.40699978953092925], [-3.6775363009592494, -3.119503850218584, 0.10699980375421318]], [[-3.6775363009596185, -2.5, -0.1174001465051065], [-3.6775363009596593, -2.5, 0.799999849793252], [-3.6775363009591624, -2.4999999999999933, 0.49999986108686345]], [[-3.677536300960194, -6.500000117923168, 2.88259994901776], [-3.6775363009603352, -6.500000122958831, 3.7999999414122003], [-3.6775363009598707, -6.500000133308523, 3.4999999999551092]]], [[[-4.962182012520562, 2.9604953405796812, -0.5104001755982058], [-4.962182015011898, 2.9604951900684804, 0.4069998139864576], [-4.962182019288785, 2.9604954085179145, 0.10699982484609351]], [[-4.9621820114691655, 3.579998929607378, -0.11740010420078348], [-4.9621820076432375, 3.579999727103961, 0.7999999016886533], [-4.962182015220621, 3.579998961529202, 0.49999990294592084]], [[-5.000000000000104, -1.5000000907833508, 2.882599962093555], [-5.0000000000000435, -1.500000091996239, 3.7999999649806995], [-5.000000000009983, -1.5000001078216592, 3.4999999999679106]]]]


Nash_values_list, Nash_points_list, probabilities_list, max_regret_list = NashEquilibriumPureStartegy(Q_game_0_0_0)
print("\n--- Nash equilibrium points of pure strategy ---")
print("Nash Q-values = ", Nash_values_list)
print("Index of Nash Q-values = ", Nash_points_list)
print("Probabilities = ", probabilities_list)
print("Maximum regret = ", max_regret_list)

Nash_values, probabilities, max_regret = NashEquilibriumMixedStartegy(Q_game_0_0_0)
print("\n--- Nash equilibrium points of mixed strategy ---")
print("Nash Q-values = ", Nash_values)
print("Probabilities = ", probabilities)
print("Maximum regret = ", max_regret)