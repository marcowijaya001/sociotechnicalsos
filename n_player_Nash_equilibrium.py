# Script for finding Nash equilibrium points for multi-player (equal or more than 3 players)

import numpy as np
import itertools


def is_best_response(payoffs, joint_action, player, n_actions):
    """Check whether `player`'s action in `joint_action` is a best response,
    holding every other player's action fixed."""
    current_value = payoffs[player][joint_action]
    for alt_action in range(n_actions):
        alt_joint = list(joint_action)
        alt_joint[player] = alt_action
        if payoffs[player][tuple(alt_joint)] > current_value:
            return False
    return True


def find_pure_nash_equilibria(payoffs, n_players, n_actions):
    """
    payoffs: list of n_players numpy arrays, each of shape (n_actions,)*n_players.
             payoffs[i][a_1,...,a_n] = player i's payoff for that joint action.
    Returns: list of joint actions (tuples) that are pure Nash equilibria.
    """
    equilibria = []
    for joint_action in itertools.product(range(n_actions), repeat=n_players):
        if all(is_best_response(payoffs, joint_action, p, n_actions) for p in range(n_players)):
            equilibria.append(joint_action)
    return equilibria


def is_global_optimum(payoffs, joint_action, n_players):
    """True if every player simultaneously achieves its highest possible
    payoff anywhere in the game at this joint action (Hu & Wellman, Def. 12)."""
    return all(
        payoffs[p][joint_action] == payoffs[p].max()
        for p in range(n_players)
    )


if __name__ == "__main__":
    # 3-player, 2-action example
    n_players, n_actions = 3, 3
    # np.random.seed(3)
    payoffs = [np.random.randint(1, 20, (n_actions,) * n_players) for _ in range(n_players)]

    print(f"{n_players}-player game, {n_actions} actions each")
    for p in range(n_players):
        print(f"\nPlayer {p + 1} payoff tensor, shape {payoffs[p].shape}:")
        print(payoffs[p])

    equilibria = find_pure_nash_equilibria(payoffs, n_players, n_actions)

    print(f"\nPure Nash equilibria found: {equilibria}")
    for ja in equilibria:
        vals = tuple(int(payoffs[p][ja]) for p in range(n_players))
        g_opt = is_global_optimum(payoffs, ja, n_players)
        print(f"  joint action {ja} -> payoffs {vals}   global optimum? {g_opt}")

    if not equilibria:
        print("  No pure-strategy equilibrium exists for this game -- "
              "a mixed-strategy solver (e.g. Gambit, sgamesolver, OpenSpiel) is needed instead.")