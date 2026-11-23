# Script for creating functions for finding Nash equilibrium points, i.e. the global optimum points and saddle points, from a Q-value matrix, for 2-player game

import numpy as np
from Game_model import *

Q_sample_2 = [
    np.array([[10,9], [0,3], [0,1]]),
    np.array([[3,0], [-1,2], [2,5]]),
    np.array([[1,1], [0,0], [0,1]]),
]

Q_sample_3 = [
    np.array([[10,9], [0,3], [0,6]]),
    np.array([[3,3], [4,2], [2,5]]),
    np.array([[1,1], [0,0], [0,6]]),
]

Q_sample_1 = [
    np.array([[10,9], [0,3]]),
    np.array([[3,0], [2,2]]),
]

Q_sample = [
    np.array([[1,0], [0,1]]),
    np.array([[0,0], [0,0]]),
]

# Function to search for Nash equilibrium points in a Q-value matrix (2 agents)
def NashEquilibriumFinder(Q_sample):

    # candidate_Nash_action_agent0 = []
    # for i in range(len(Q_sample[0])): # number of column
    #     value_agent0 = []
    #     candidate_Nash_agent0 = []
    #     for j in range(len(Q_sample)): # number of row
    #         value_agent0.append(Q_sample[j][i][0])
    #         # print(value_agent0) # for testing
    #     candidate_Nash_agent0.append([np.max(value_agent0), np.argmax(value_agent0), i])  # Starting point of search is agent 1 (column-wise)
    #     # Create a function to return identical maximum value to be multiple Nash equilibrium points
    #     # print(candidate_Nash_agent0) # for testing
    #     Q_value = candidate_Nash_agent0[0][0]
    #     p = candidate_Nash_agent0[0][1]  # row index of candidate Nash
    #     q = candidate_Nash_agent0[0][2]  # column index of candidate Nash
    #     candidate_Nash_action_agent0.append([Q_value, p, q])
    # print("Candidate Nash action of agent 0: ", candidate_Nash_action_agent0)

    # candidate_Nash_action_agent1 = []
    # for i in range(len(Q_sample)): # number of row
    #     value_agent1 = []
    #     candidate_Nash_agent1 = []
    #     for j in range(len(Q_sample[0])): # number of column
    #         value_agent1.append(Q_sample[i][j][1])
    #     candidate_Nash_agent1.append([np.max(value_agent1), i, np.argmax(value_agent1)]) # Starting point of search is agent 1 (column-wise)
    #     # print(candidate_Nash_agent1)
    #     Q_value = candidate_Nash_agent1[0][0]
    #     p = candidate_Nash_agent1[0][1] # row index of candidate Nash
    #     q = candidate_Nash_agent1[0][2] # column index of candidate Nash
    #     candidate_Nash_action_agent1.append([Q_value, p, q])
    # #print("Candidate Nash action of agent 1: ", candidate_Nash_action_agent1)

    # Updated function to find candidate of Nash equilibrium points per agent
    candidate_Nash_action_agent0 = []
    for col in range(len(Q_sample[0])):
        value_agent0 = np.array([Q_sample[row][col][0] for row in range(len(Q_sample))])
        max_value_agent0 = np.max(value_agent0)
        max_rows_agent0 = np.where(value_agent0 == max_value_agent0)[0]

        for row in max_rows_agent0:
            candidate_Nash_action_agent0.append([max_value_agent0, row, col])

    candidate_Nash_action_agent1 = []
    for row in range(len(Q_sample)):
        value_agent1 = np.array([Q_sample[row][col][1] for col in range(len(Q_sample[0]))])
        max_value_agent1 = np.max(value_agent1)
        max_cols_agent1 = np.where(value_agent1 == max_value_agent1)[0]

        for col in max_cols_agent1:
            candidate_Nash_action_agent1.append([max_value_agent1, row, col])

    Nash_equilibrium_points_list = []
    Nash_Q_values_list = []
    Nash_equilibrium_list = []
    for n in range(len(candidate_Nash_action_agent0)):
        for m in range(len(candidate_Nash_action_agent1)):
            if (candidate_Nash_action_agent0[n][1] == candidate_Nash_action_agent1[m][1] and
                    candidate_Nash_action_agent0[n][2] == candidate_Nash_action_agent1[m][2]):
                Nash_equilibrium_points = [candidate_Nash_action_agent0[n][1], candidate_Nash_action_agent0[n][2]]
                Nash_equilibrium_points_list.append(Nash_equilibrium_points)
                #print("A Nash equilibrium point is found at action: ", Nash_equilibrium_points)
                Nash_Q_values = [candidate_Nash_action_agent0[n][0], candidate_Nash_action_agent1[m][0]]
                Nash_Q_values_list.append(Nash_Q_values)

                Nash_equilibrium =  [[candidate_Nash_action_agent0[n][0], candidate_Nash_action_agent1[m][0]], candidate_Nash_action_agent0[n][1], candidate_Nash_action_agent0[n][2]]
                Nash_equilibrium_list.append(Nash_equilibrium)

    return Nash_equilibrium_list, Nash_equilibrium_points_list, Nash_Q_values_list

# Nash_equilibrium_list, Nash_equilibrium_points_list, Nash_Q_values_list = NashEquilibriumFinder(Q_sample_3)
# print("List of Nash equilibrium points: ", Nash_equilibrium_list)
# print("List of Nash equilibrium actions: ", Nash_equilibrium_points_list)
# print("List of Nash Q-values", Nash_Q_values_list)


# Function to search for global optimum in a Q-value matrix (2-agents)
def GlobalOptimumPoint(Q_sample):

    # Global optimum value of agent 1
    candidate_max_action_agent1 = []
    Q_value_list_agent1 = []
    for i in range(len(Q_sample)):
        value_agent1 = []
        candidate_max_agent1 = []
        for j in range(len(Q_sample[0])):
            value_agent1.append(Q_sample[i][j][1])
        candidate_max_agent1.append([np.max(value_agent1), i, np.argmax(value_agent1)])
        Q_value = candidate_max_agent1[0][0]
        Q_value_list_agent1.append(Q_value)
        p = candidate_max_agent1[0][1]  # row index of candidate Nash
        q = candidate_max_agent1[0][2]  # column index of candidate Nash
        candidate_max_action_agent1.append([Q_value, p, q])
    Q_value_max_agent1 = np.max(Q_value_list_agent1)
    global_optimum_agent1_list = []
    for a in range(len(candidate_max_action_agent1)):
        if candidate_max_action_agent1[a][0] == Q_value_max_agent1:
            global_optimum_agent1 = [candidate_max_action_agent1[a][0], candidate_max_action_agent1[a][1], candidate_max_action_agent1[a][2]]
            global_optimum_agent1_list.append(global_optimum_agent1)

    # Global optimum value of agent 0
    candidate_max_action_agent0 = []
    Q_value_list_agent0 = []
    for i in range(len(Q_sample[0])):
        value_agent0 = []
        candidate_max_agent0 = []
        for j in range(len(Q_sample)):
            value_agent0.append(Q_sample[j][i][0])
        candidate_max_agent0.append([np.max(value_agent0), np.argmax(value_agent0), i])
        Q_value = candidate_max_agent0[0][0]
        Q_value_list_agent0.append(Q_value)
        p = candidate_max_agent0[0][1]  # row index of candidate Nash
        q = candidate_max_agent0[0][2]  # column index of candidate Nash
        candidate_max_action_agent0.append([Q_value, p, q])
    Q_value_max_agent0 = np.max(Q_value_list_agent0)
    global_optimum_agent0_list = []
    for a in range(len(candidate_max_action_agent0)):
        if candidate_max_action_agent0[a][0] == Q_value_max_agent0:
            global_optimum_agent0 = [candidate_max_action_agent0[a][0], candidate_max_action_agent0[a][1], candidate_max_action_agent0[a][2]]
            global_optimum_agent0_list.append(global_optimum_agent0)

    # Determining global optimum point by combining both global optimum of agent 0 and agent 1
    global_optimum_point_list = []
    for i in range(len(global_optimum_agent0_list)):
        for j in range(len(global_optimum_agent1_list)):
            if global_optimum_agent0_list[i][1] == global_optimum_agent1_list[j][1] and global_optimum_agent0_list[i][2] == global_optimum_agent1_list[j][2]:
                global_optimum_point = [[global_optimum_agent0_list[i][0], global_optimum_agent1_list[j][0]], global_optimum_agent0_list[i][1], global_optimum_agent0_list[j][2]]
                global_optimum_point_list.append(global_optimum_point)
    if len(global_optimum_point_list) == 0:
        print("There is no global optimum point.")

    return global_optimum_agent1_list, global_optimum_agent0_list, global_optimum_point_list

# a, b, c = GlobalOptimumPoint(Q_sample)
# print("Global optimum point of agent 1: ", a)
# print("Global optimum point of agent 2: ", b)
# print("List of global optimum points: ", c)


def IsSaddlePoint(Q_sample):

    # Criteria 0: a saddle point must be a Nash equilibrium point
    a, b, c =NashEquilibriumMixedStrategy(Q_sample)

    # Criteria 1: determining if Q-value of agent0 is the minimum across all columns in the same row
    candidate_saddle_point_agent0 = []
    for i in range(len(a)):
        column_value_list = []
        for j in range(len(Q_sample[0])):
            column_value = Q_sample[a[i][1]][j][0]
            column_value_list.append(column_value)
        # print(column_value_list)
        min_column_value = np.min(column_value_list)
        if min_column_value == a[i][0][0]:
            candidate_saddle_point_agent0.append(a[i])
        else:
            pass

    # Criteria 2: determining if Q-value of agent1 is the minimum across all rows in the same column
    candidate_saddle_point_agent1 = []
    for i in range(len(a)):
        row_value_list = []
        for j in range(len(Q_sample)):
            row_value = Q_sample[j][a[i][2]][1] # fix this
            row_value_list.append(row_value)
        # print(row_value_list)
        min_row_value = np.min(row_value_list)
        if min_row_value == a[i][0][1]:
            candidate_saddle_point_agent1.append(a[i])
        else:
            pass

    # If fulfill all criteria, then it is a saddle point
    saddle_point_list = []
    for p in range(len(candidate_saddle_point_agent0)):
        for q in range(len(candidate_saddle_point_agent1)):
            if candidate_saddle_point_agent0[p][1] == candidate_saddle_point_agent1[q][1] and candidate_saddle_point_agent0[p][2] == candidate_saddle_point_agent1[q][2]:
                saddle_point = [candidate_saddle_point_agent0[p][0], candidate_saddle_point_agent0[p][1], candidate_saddle_point_agent0[p][2]]
                saddle_point_list.append(saddle_point)
    if len(saddle_point_list) == 0:
        print("There is no saddle point.")

    return candidate_saddle_point_agent0, candidate_saddle_point_agent1, saddle_point_list

# r, s, t = IsSaddlePoint(Q_sample)
# print("Candidate saddle point of agent 0: ", r)
# print("Candidate saddle point of agent 1: ", s)
# print("List of saddle points: ", t)


