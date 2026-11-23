import warnings
from itertools import combinations, product

import numpy as np
from scipy.optimize import least_squares


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
# Mixed strategies
# ---------------------------------------------------------------------------
# A mixed strategy of agent i is a probability vector p_i over its n_i actions.
# A mixed profile is a list sigma = [p_0, p_1, ..., p_{N-1}].
# A pure action is the special case of a one-hot vector, e.g. [0, 1].
# ---------------------------------------------------------------------------
def _contract_others(T, sigma, keep=None):
    """Average tensor T (one axis per agent) over every agent's mixed strategy,
    except agent `keep`. Returns a vector over agent keep's actions, or a scalar
    if keep is None."""
    for k in reversed(range(len(sigma))):      # back to front keeps axis numbers valid
        if k != keep:
            T = np.tensordot(T, sigma[k], axes=([k], [0]))
    return T


def action_values(Q_sample, sigma, i):
    """Expected value of each pure action of agent i, given that the other
    agents play their mixed strategies in sigma."""
    Q = to_payoff_tensor(Q_sample)
    return _contract_others(Q[..., i], sigma, keep=i)


def expected_values(Q_sample, sigma):
    """Expected value [q0, q1, ...] of every agent under mixed profile sigma."""
    Q = to_payoff_tensor(Q_sample)
    return np.array([float(_contract_others(Q[..., i], sigma)) for i in range(Q.shape[-1])])


def nash_regret(Q_sample, sigma):
    """For each agent: how much it could gain by switching to its best pure
    action while the others keep playing sigma. sigma is a Nash equilibrium
    if and only if every entry is 0 (up to rounding)."""
    Q = to_payoff_tensor(Q_sample)
    sigma = [np.asarray(p, dtype=float) for p in sigma]
    regret = []
    for i in range(Q.shape[-1]):
        av = _contract_others(Q[..., i], sigma, keep=i)
        regret.append(av.max() - av @ sigma[i])
    return np.array(regret)


def _remove_dominated(Q):
    """Iteratively remove pure actions that are strictly dominated by another
    pure action. No Nash equilibrium (pure or mixed) ever uses such an action,
    so this only shrinks the search. Returns the kept action indices per agent."""
    N = Q.shape[-1]
    keep = [list(range(n_i)) for n_i in Q.shape[:-1]]
    changed = True
    while changed:
        changed = False
        for i in range(N):
            sub = np.moveaxis(Q[np.ix_(*keep)][..., i], i, 0)   # own actions first
            sub = sub.reshape(len(keep[i]), -1)
            for a in range(len(keep[i])):
                if any(np.all(sub[b] > sub[a]) for b in range(len(keep[i])) if b != a):
                    del keep[i][a]
                    changed = True
                    break
            if changed:
                break
    return keep


def _conditionally_dominated(Q, S):
    """True if some action in a support is strictly dominated when the other
    agents are restricted to their supports: that support cannot be an equilibrium."""
    N = Q.shape[-1]
    for i in range(N):
        idx = [list(S[k]) if k != i else list(range(Q.shape[i])) for k in range(N)]
        sub = np.moveaxis(Q[np.ix_(*idx)][..., i], i, 0).reshape(Q.shape[i], -1)
        for a in S[i]:
            if np.any(np.all(sub > sub[a], axis=1)):
                return True
    return False


def _solve_support(Q, S, n_starts, rng):
    """Find mixed profiles whose support is S = (S_0, ..., S_{N-1}) and where
    every agent is indifferent between all actions in its support.

    Unknowns: for each agent, the probabilities of all but the last action in
    its support (the last one is 1 minus the rest). Equations: equal expected
    value for every action in the support. Same number of equations as unknowns.
    """
    N, n = len(S), Q.shape[:-1]
    sizes = [len(s) for s in S]
    free = [k - 1 for k in sizes]
    offsets = np.cumsum([0] + free)

    def unpack(x):
        sigma = []
        for i in range(N):
            head = x[offsets[i]:offsets[i + 1]]
            p = np.zeros(n[i])
            p[list(S[i])] = np.append(head, 1.0 - head.sum())
            sigma.append(p)
        return sigma

    def residual(x):
        sigma = unpack(x)
        r = []
        for i in range(N):
            av = _contract_others(Q[..., i], sigma, keep=i)[list(S[i])]
            r.extend(av[1:] - av[0])
        return np.array(r)

    if sum(free) == 0:                                  # pure profile, nothing to solve
        return [unpack(np.zeros(0))]

    # For 2 agents the equations are linear, so one start is enough.
    # For 3+ agents they are polynomial: use several starts.
    n_starts = 1 if N == 2 else n_starts
    starts = [np.concatenate([np.full(k - 1, 1.0 / k) for k in sizes])]
    starts += [np.concatenate([rng.dirichlet(np.ones(k))[:-1] for k in sizes])
               for _ in range(n_starts - 1)]

    solutions = []
    for x0 in starts:
        res = least_squares(residual, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.max(np.abs(res.fun)) > 1e-8:
            continue
        sigma = unpack(res.x)
        if min(p.min() for p in sigma) < -1e-9:         # not a probability vector
            continue
        sigma = [np.clip(p, 0.0, None) for p in sigma]
        solutions.append([p / p.sum() for p in sigma])
    return solutions


def MixedNashEquilibriumFinder(Q_sample, find="all", n_starts=10, tol=1e-6, seed=0):
    """Mixed-strategy Nash equilibria of an N-agent game (support enumeration).

    Every finite game has at least one mixed Nash equilibrium (Nash, 1950), so
    this finds something even when NashEquilibriumFinder returns nothing.
    Pure equilibria show up as one-hot strategies.

    Method: remove strictly dominated actions, then try every combination of
    supports (the set of actions each agent plays with positive probability),
    smallest first. For each, solve "every agent is indifferent between the
    actions in its support", then keep the solution only if nash_regret
    confirms it is an equilibrium. Every returned profile is therefore a
    verified equilibrium.

    find      "all" (every equilibrium found) or "first" (stop at the first one)
    n_starts  starting points per support for 3+ agents (2 agents: exact)
    tol       allowed regret, relative to the payoff range

    Returns (same structure as NashEquilibriumFinder, with probability vectors
    in place of action indices):
      Mixed_equilibrium_list   [[q0, q1, q2], p0, p1, p2]
      Mixed_strategies_list    [p0, p1, p2]
      Mixed_Q_values_list      [q0, q1, q2]   (expected values)
    where p_i is agent i's probability vector over all its actions.
    """
    if find not in ("all", "first"):
        raise ValueError("find must be 'all' or 'first'")
    Q = to_payoff_tensor(Q_sample).astype(float)
    n = Q.shape[:-1]
    scale = max(1.0, float(np.ptp(Q)))
    rng = np.random.default_rng(seed)

    keep = _remove_dominated(Q)
    Qr = Q[np.ix_(*keep)]                               # reduced game
    supports = [[s for k in range(1, len(kp) + 1) for s in combinations(range(len(kp)), k)]
                for kp in keep]
    # small and balanced supports first (Porter, Nudelman & Shoham, 2004)
    profiles = sorted(product(*supports),
                      key=lambda S: (sum(map(len, S)), max(map(len, S)) - min(map(len, S))))

    found = []
    for S in profiles:
        if _conditionally_dominated(Qr, S):
            continue
        for sigma_r in _solve_support(Qr, S, n_starts, rng):
            sigma = []                                  # back to the full action sets
            for i, kp in enumerate(keep):
                p = np.zeros(n[i])
                p[kp] = sigma_r[i]
                sigma.append(p)
            if nash_regret(Q, sigma).max() > tol * scale:
                continue
            duplicate = any(all(np.allclose(p, q, atol=1e-6) for p, q in zip(sigma, f))
                            for f in found)
            if not duplicate:
                found.append(sigma)
        if find == "first" and found:
            break

    if not found:
        warnings.warn("No equilibrium found. One always exists: increase n_starts "
                      "or cross-check with Gambit (pip install pygambit).")

    Mixed_equilibrium_list, Mixed_strategies_list, Mixed_Q_values_list = [], [], []
    for sigma in found:
        sigma = [np.where(np.abs(p) < 1e-12, 0.0, p) for p in sigma]
        q = expected_values(Q, sigma).tolist()
        p_list = [p.tolist() for p in sigma]
        Mixed_equilibrium_list.append([q, *p_list])
        Mixed_strategies_list.append(p_list)
        Mixed_Q_values_list.append(q)

    return Mixed_equilibrium_list, Mixed_strategies_list, Mixed_Q_values_list


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

    Q_sample = np.array([
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

    Nash_equilibrium_list, Nash_equilibrium_points_list, Nash_Q_values_list = \
        NashEquilibriumFinder(Q_3p)
    print("List of Nash equilibrium points: ", Nash_equilibrium_list)
    print("List of Nash equilibrium actions: ", Nash_equilibrium_points_list)
    print("List of Nash Q-values", Nash_Q_values_list)

    per_agent, global_points = GlobalOptimumPoint(Q_3p)
    for i, g in enumerate(per_agent):
        print(f"Global optimum point(s) of agent {i}: ", g)
    print("List of global optimum points: ", global_points)

    candidates, saddle_points = IsSaddlePoint(Q_3p)
    print("List of saddle points: ", saddle_points)

    # Mixed equilibria of the same game: the pure one appears as one-hot vectors
    Mixed_equilibrium_list, _, _ = MixedNashEquilibriumFinder(Q_3p)

    print("\nMixed Nash equilibria [[q0, q1, q2], p0, p1, p2]:")
    for eq in Mixed_equilibrium_list:
        print("  ", np.round(eq[0], 4).tolist(), *[np.round(p, 4).tolist() for p in eq[1:]])

    # -----------------------------------------------------------------------
    # A 3-agent game with NO pure Nash equilibrium (3-player matching pennies):
    # agent 0 wants to match agent 1, agent 1 wants to match agent 2,
    # agent 2 wants to mismatch agent 0. Every pure joint action leaves
    # somebody wanting to switch.
    # -----------------------------------------------------------------------
    Q_pennies = np.zeros((2, 2, 2, 3))
    for a0, a1, a2 in product(range(2), repeat=3):
        Q_pennies[a0, a1, a2] = [1 if a0 == a1 else -1,
                                 1 if a1 == a2 else -1,
                                 1 if a2 != a0 else -1]

    print("\n3-player matching pennies")
    print("Pure Nash equilibria: ", NashEquilibriumFinder(Q_pennies)[1])
    eq_list, strategies, values = MixedNashEquilibriumFinder(Q_pennies)
    print("Mixed Nash equilibria:", eq_list)
    print("Regret check (all ~0):", nash_regret(Q_pennies, strategies[0]))