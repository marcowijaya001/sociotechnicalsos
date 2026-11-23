import numpy as np


# ---------------------------------------------------------------------------
# Input format
# ---------------------------------------------------------------------------
# A game with N agents is a numpy array of shape (n_0, n_1, ..., n_{N-1}, N):
#
#     Q[a0, a1, a2] = [q_agent0, q_agent1, q_agent2]      (3 agents)
#
# where a_i is the action index of agent i and n_i its number of actions.
# The old 2-agent format (a list of arrays of shape (n_1, 2)) is the same
# idea with N = 2, so it still works unchanged.
#
# Two helpers below build this array from more convenient formats.
# ---------------------------------------------------------------------------


def to_payoff_tensor(Q_sample):
    """Convert input to an array of shape (n_0, ..., n_{N-1}, N) and check it."""
    Q = np.asarray(Q_sample)
    n_agents = Q.shape[-1]
    if Q.ndim - 1 != n_agents:
        raise ValueError(
            f"Q has shape {Q.shape}: the last axis says {n_agents} agents, "
            f"but there are {Q.ndim - 1} action axes. Expected shape "
            f"(n_0, ..., n_{{N-1}}, N)."
        )
    return Q


def from_agent_tables(*tables):
    """One Q-table per agent, each of shape (n_0, n_1, n_2) -> joint tensor.

    Example: Q = from_agent_tables(Q0, Q1, Q2)
    where Q0[a0, a1, a2] is agent 0's value of joint action (a0, a1, a2).
    """
    return np.stack([np.asarray(t) for t in tables], axis=-1)


def from_layers(layers):
    """Textbook layout: one 2D matrix per action of agent 2.

    layers[a2][a0][a1] = [q0, q1, q2]
    i.e. agent 0 picks the row, agent 1 the column, agent 2 the matrix.
    """
    L = np.asarray(layers)          # (n_2, n_0, n_1, 3)
    return np.moveaxis(L, 0, 2)     # (n_0, n_1, n_2, 3)


def _idx(a):
    return tuple(int(x) for x in a)


# ---------------------------------------------------------------------------
# Nash equilibrium (pure strategies)
# ---------------------------------------------------------------------------
def NashEquilibriumFinder(Q_sample, atol=0.0):
    """Pure-strategy Nash equilibria of an N-agent game.

    A joint action is a Nash equilibrium if every agent's value there is the
    maximum over its own actions, with the other agents' actions held fixed.
    Use atol > 0 to treat near-ties as ties (useful for learned Q-values).

    Returns (same structure as the 2-agent version):
      Nash_equilibrium_list        [[q0, q1, q2], a0, a1, a2]
      Nash_equilibrium_points_list [a0, a1, a2]
      Nash_Q_values_list           [q0, q1, q2]
    """
    Q = to_payoff_tensor(Q_sample)
    n_agents = Q.shape[-1]

    is_nash = np.ones(Q.shape[:-1], dtype=bool)
    for i in range(n_agents):
        qi = Q[..., i]
        best_reply = qi.max(axis=i, keepdims=True)   # max over agent i's own actions
        is_nash &= qi >= best_reply - atol

    Nash_equilibrium_list, Nash_equilibrium_points_list, Nash_Q_values_list = [], [], []
    for a in np.argwhere(is_nash):
        point = [int(x) for x in a]
        q = Q[_idx(a)].tolist()
        Nash_equilibrium_list.append([q, *point])
        Nash_equilibrium_points_list.append(point)
        Nash_Q_values_list.append(q)

    return Nash_equilibrium_list, Nash_equilibrium_points_list, Nash_Q_values_list


# ---------------------------------------------------------------------------
# Global optimum
# ---------------------------------------------------------------------------
def GlobalOptimumPoint(Q_sample):
    """Joint actions where each agent reaches its own global maximum value.

    Returns:
      global_optimum_per_agent  list indexed by agent i; each entry is a list of
                                [q_i, a0, a1, a2] at agent i's global maximum
      global_optimum_point_list [[q0, q1, q2], a0, a1, a2] where all agents are
                                at their global maximum simultaneously
    """
    Q = to_payoff_tensor(Q_sample)
    n_agents = Q.shape[-1]

    global_optimum_per_agent = []
    all_at_max = np.ones(Q.shape[:-1], dtype=bool)
    for i in range(n_agents):
        qi = Q[..., i]
        at_max = qi == qi.max()
        all_at_max &= at_max
        global_optimum_per_agent.append(
            [[qi[_idx(a)].item(), *[int(x) for x in a]] for a in np.argwhere(at_max)]
        )

    global_optimum_point_list = [
        [Q[_idx(a)].tolist(), *[int(x) for x in a]] for a in np.argwhere(all_at_max)
    ]
    if len(global_optimum_point_list) == 0:
        print("There is no global optimum point.")

    return global_optimum_per_agent, global_optimum_point_list


# ---------------------------------------------------------------------------
# Saddle point
# ---------------------------------------------------------------------------
def IsSaddlePoint(Q_sample, mode="joint"):
    """Saddle points: Nash equilibria where each agent's value is also the
    minimum over the other agents' actions, with its own action held fixed.

    mode="joint":      minimum over ALL combinations of the other agents'
                       actions (stricter).
    mode="unilateral": minimum over each single other agent deviating alone.
    For 2 agents the two modes give the same result as the original code.

    Returns:
      candidate_saddle_point_per_agent  list indexed by agent i; Nash points
                                        that satisfy the min-criterion for agent i
      saddle_point_list                 [[q0, q1, q2], a0, a1, a2] satisfying it
                                        for all agents
    """
    if mode not in ("joint", "unilateral"):
        raise ValueError("mode must be 'joint' or 'unilateral'")

    Q = to_payoff_tensor(Q_sample)
    n_agents = Q.shape[-1]

    # Criterion 0: a saddle point must be a Nash equilibrium point
    nash_list, _, _ = NashEquilibriumFinder(Q)

    candidate_saddle_point_per_agent = [[] for _ in range(n_agents)]
    saddle_point_list = []
    for eq in nash_list:
        q, a = eq[0], eq[1:]
        is_saddle = True
        for i in range(n_agents):
            if mode == "joint":
                # fix agent i's action, let all other agents vary
                sl = tuple(a[k] if k == i else slice(None) for k in range(n_agents))
                ok = q[i] <= Q[sl + (i,)].min()
            else:
                ok = True
                for j in range(n_agents):
                    if j == i:
                        continue
                    # only agent j varies, everyone else stays at the equilibrium
                    sl = tuple(slice(None) if k == j else a[k] for k in range(n_agents))
                    ok = ok and q[i] <= Q[sl + (i,)].min()
            if ok:
                candidate_saddle_point_per_agent[i].append(eq)
            is_saddle = is_saddle and ok
        if is_saddle:
            saddle_point_list.append(eq)

    if len(saddle_point_list) == 0:
        print("There is no saddle point.")

    return candidate_saddle_point_per_agent, saddle_point_list


# ---------------------------------------------------------------------------
# Example: 3-agent game, 2 actions each
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Option A: write the tensor directly, Q_3p[a0][a1][a2] = [q0, q1, q2]
    Q_3p = np.array([
        [  # a0 = 0
            [[3, 3, 3], [0, 4, 0]],   # a1 = 0 : a2 = 0, a2 = 1
            [[0, 0, 4], [1, 1, 1]],   # a1 = 1
        ],
        [  # a0 = 1
            [[4, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2]],
        ],
    ])

    # Option B: same game written as one matrix per action of agent 2
    Q_3p_layers = from_layers([
        # a2 = 0: rows = a0, columns = a1
        [[[3, 3, 3], [0, 0, 4]],
         [[4, 0, 0], [1, 1, 1]]],
        # a2 = 1
        [[[0, 4, 0], [1, 1, 1]],
         [[1, 1, 1], [2, 2, 2]]],
    ])
    assert np.array_equal(Q_3p, Q_3p_layers)

    Q_sample = from_layers([
        # a2 = 0: rows = a0, columns = a1
        [[[10, 3, 7], [0, 4, 4]],
         [[3, 2, 4], [2, 9, 2]]],
        # a2 = 1
        [[[0, 1, 2], [3, 3, 1]],
         [[5, 0, 1], [2, 5, 3]]],
    ])

    # Option C: same game built from one Q-table per agent
    Q0, Q1, Q2 = Q_3p[..., 0], Q_3p[..., 1], Q_3p[..., 2]
    assert np.array_equal(Q_3p, from_agent_tables(Q0, Q1, Q2))

    Nash_equilibrium_list, Nash_equilibrium_points_list, Nash_Q_values_list = NashEquilibriumFinder(Q_sample)
    print("List of Nash equilibrium points: ", Nash_equilibrium_list)
    print("List of Nash equilibrium actions: ", Nash_equilibrium_points_list)
    print("List of Nash Q-values", Nash_Q_values_list)

    per_agent, global_points = GlobalOptimumPoint(Q_sample)
    for i, g in enumerate(per_agent):
        print(f"Global optimum point(s) of agent {i}: ", g)
    print("List of global optimum points: ", global_points)

    candidates, saddle_points = IsSaddlePoint(Q_sample)
    print("List of saddle points: ", saddle_points)