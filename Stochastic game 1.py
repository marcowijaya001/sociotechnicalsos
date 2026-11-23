import sgamesolver
import numpy as np

payoff_matrices = [  # state0:
                   np.array([[[0, -1,  1],
                              [1,  0, -1],
                              [-1, 1,  0]],
                             [[0,  1, -1],
                              [-1, 0,  1],
                              [1, -1,  0]]]),
                     # state1:
                   np.array([[[0, -1,  1],
                              [1,  0, -1],
                              [-1, 1,  1]],    # player0 wins tie on Scissors
                             [[0,  1, -1],
                              [-1, 0,  1],
                              [1, -1, -1]]]),  # player1 loses tie on Scissors
                     # state2:
                   np.array([[[0, -1,  1],
                              [1,  0, -1],
                              [-1, 1, -1]],    # player0 loses tie on Scissors
                             [[0,  1, -1],
                              [-1, 0,  1],
                              [1, -1,  1]]])]  # player1 wins tie on Scissors

# transitions identical for each state
transition_matrices = [np.array([[[1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0]]])] * 3

common_discount_factor = 0.95

game = sgamesolver.SGame(payoff_matrices=payoff_matrices,
                        transition_matrices=transition_matrices,
                        discount_factors=common_discount_factor)

# game = sgamesolver.SGame.random_game(3, 3, 3, seed=100)
homotopy = sgamesolver.homotopy.LogTracing(game)
homotopy.solver_setup()
homotopy.solver.start_storing_path()
homotopy.solver.verbose = 4
homotopy.solve()
homotopy.plot_path()