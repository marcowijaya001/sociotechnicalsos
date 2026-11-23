# The script for experimenting games

import numpy as np
from Nash_equilibrium import NashEquilibriumMixedStrategy

# Initial state: Game_0_0 -> (planning; low FoMs)
Q_Game_0_0 = [np.array([[36.62146657, 20.43410964], [36.62146657, 18.43410964], [ 0.        , -4.        ]]),
            np.array([[46.68657665, 24.35160445], [50.74960191,  3.        ], [-5.        ,  1.        ]])]

# Transition to: Game_1_0 -> (ownership of S1; low FoMs)
Q_Game_1_0 = [np.array([[51.68657665, 19.35160445], [55.74960191, 20.29984077], [ 0.        , -4.        ]]),
              np.array([[57.68657665, 22.35160445], [61.74960191, 23.29984077], [20.8203    ,  5.91614   ]])]

# Transition to: Game_2_1 -> (operation; medium FoMs)
Q_Game_2_1 =[np.array([[65.74960191, 26.29984077], [24.8203    ,  8.91614   ], [10.        ,  4.        ]]),
             np.array([[47.87978613, 30.35191445], [-8.        ,  6.        ], [-8.        ,  8.        ]]),
             np.array([[ 0.,  0.], [ 0., -2.], [ 0.,  0.]])]

# Transition to: Game_2_3 (END) -> (operation; end of development)

# Result after 500 iterations, stopping criteria is if either or both agent(s) reach "END" state(s):
## Agent 0: planning -> ownership of S1 -> operation   -> operation
## Agent 1: low FoMs -> low FoMs        -> medium FoMs -> end of development


# TO DO (possibilities):
## 1) Convergence criteria and make plots of Q-values,
## 2) Exploration of reward functions/structure to see different sequence/optimal behavior of agents
## 3) Scale up the game size to involving 3 or more agents -> generalize the formulation to find Nash equilibrium points
## 4) Add stochasticity to the game transitions (conceptually done)
## 5) Varying the use of Nash equilibrium for each game (fist, second, or last found), hence different Nash Q-values and possible different optimal solution


# Experiment 1 : 2000 iterations
## Initial state: Game_0_0
Exp1_Q_Game_0_0 = [np.array([[49.70966294, 25.52025942], [49.70966294, 23.52025942], [ 0.        , -4.        ]]),
                   np.array([[59.61910837, 29.47738679], [63.60486468,  3.        ], [-5.        ,  1.        ]])]
# Transition to: Game_1_0
Exp1_Q_Game_1_0 = [np.array([[64.61910837, 24.47738679], [68.60486468, 25.44194587], [ 0.        , -4.        ]]),
                  np.array([[70.61910837, 27.47738679], [74.60486468, 28.44194587], [20.842575  ,  5.926535  ]])]
## Transition to: Game_2_1
Exp1_Q_Game_2_1 = [np.array([[78.60486468, 31.44194587], [24.842575  ,  8.926535  ], [10.        ,  4.        ]]),
                   np.array([[60.64377409, 35.45750964], [-8.        ,  6.        ], [-8.        ,  8.        ]]),
                   np.array([[ 0.,  0.],                 [ 0., -2.],                 [ 0.,  0.]])]
## Transition to: Game_2_1 (stuck in same game)
## Result after 2000 iterations:
### Agent 0: planning -> ownership of S1 -> operation   -> operation
### Agent 1: low FoMs -> low FoMs        -> medium FoMs -> medium FoMs

### The case where more/non-zero rewards are given to end/move from the same state
Exp1_Q_Game_2_1_plus = [np.array([[78.60486468, 31.44194587], [24.842575  ,  8.926535  ], [10.        ,  4.        ]]),
                   np.array([[ 60.64377409,  35.45750964], [ -8.        ,   6.        ], [ 91.        , 107.        ]]),
                   np.array([[ 0.,  0.], [ 0., -2.], [ 0.,  0.]])]
### Transition to: Game_1_3 (END); both rewards=...27 -> stay, both rewards=28,29,30,... -> 2 Nash equilibrium points (stay and end); reward agent 0=20 and agent1=100 -> 2 Nash equilibrium points, (10,100) -> stay
## Result after 2000 iterations:
### Agent 0: planning -> ownership of S1 -> operation   -> ownership of S1
### Agent 1: low FoMs -> low FoMs        -> medium FoMs -> end of development

Q_sample_0 = [np.array([[42.60179748, 15.80290507], # From Game_1_0: pure strategy
       [46.2417047 , 16.49668188],
       [ 0.        , -4.        ]]), np.array([[48.60179748, 18.80290507],
       [52.2417047 , 19.49668188],
       [19.49325   ,  5.29685   ]])]

Q_sample_1 = [np.array([[39.22077381, 15.2051138 ], # From Game_1_0: mixed strategy, with stochasticity
       [46.2417047 , 16.49668188],
       [ 0.        , -4.        ]]), np.array([[48.60179748, 18.80290507],
       [39.13962282, 14.61548913],
       [19.49325   ,  5.29685   ]])]

Q_sample_2 = [np.array([[ 0,  0],
       [ 0,  0],
       [ 0, -4]]), np.array([[ 1,  6],
       [ 8,  2],
       [-4,  1]])]

Nash_equilibrium_list, Nash_equilibrium_points_list, Nash_Q_values_list = NashEquilibriumMixedStrategy(Q_sample_1)
print("List of Nash equilibrium points: ", Nash_equilibrium_list)