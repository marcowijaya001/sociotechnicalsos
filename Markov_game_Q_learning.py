"""
Nash Q-LEARNING for the SoS1 Markov game, learned from Monte Carlo sequences of stage games.

Difference with the Q-value ITERATION in SoS_Markov_game_with_transitions.py:
    iteration : every stage game and every joint action is updated in each sweep, using the
                known transition probabilities (expected next value)
    learning  : only the (stage game, joint action) pairs that occur in simulated sequences are
                updated, using the next stage game that actually happened (sampled transition)

Update for every step (s, a, s') of every simulated sequence, for every agent i:
    Q_s[a][i] <- (1 - alpha) * Q_s[a][i] + alpha * (r_s[a][i] + beta * NashQ_s'[i])
    alpha = 1 / (n + 1),  n = number of times stage game s has been updated before
    NashQ_s' = Nash value of stage game s' from its current Q values
               (same solver as the generated script: pygambit logit_solve)

Needs SoS_Markov_game_with_transitions.py in the same folder.
"""
import json
import time

import numpy as np

from SoS_markov_game_MonteCarlo_automated import (
    MarkovGame,
    Trajectory,
    NashEquilibriumMixedStrategyProbs,    # same pygambit solver as NashEquilibriumMixedStrategy,
)                                         # also returns the strategies (for the best policy)


# ------------ Start of SoS1 use case ------------ #

def system_acquisition_cost(s, a):
    # acquisition cost depends on system's FoM level the private company is producing
    return 0 if s["Private company"].name == "Low Battery, Low Satellite #2" else 0

def tech_acquisition_cost(s, a):
    # acquisition cost depends on technology's FoM level the private company is producing
    return 0 if s["Private company"].name == ["Low Battery, Low Satellite #2", "Low Battery, High Satellite #2"] else 0

SoS1 = {
    "stakeholders": [
        {
            "id": "Ministry of Finance",
            "states": {
                "Policy Making": [],
                "Funding #1": [],
                "Funding #2": [],
                "Data Collection": [],
                "End of Use": [],
            },
            "actions": ["stay", "fund", "use", "end"],
            "available_actions": {
                "Policy Making": ["stay", "fund"],
                "Funding #1": ["stay", "fund"],
                "Funding #2": ["stay", "use"],
                "Data Collection": ["use", "end"],
                "End of Use": ["stay"],            # added: terminal state (otherwise ALL actions are available)
            },
            "own_rewards": {
                "Policy Making": {"stay": -1, "fund": 0},
                "Funding #1": {"stay": -1, "fund": 0},
                "Funding #2": {"stay": -1, "use": 2},
                "Data Collection": {"use": 2, "end": 0},
            },
            "transitions": [
                {"initial_state": "Policy Making", "action": "stay", "final_state": "Policy Making"},
                {"initial_state": "Policy Making", "action": "fund", "final_state": "Funding #1"},
                {"initial_state": "Funding #1", "action": "stay", "final_state": "Funding #1"},
                {"initial_state": "Funding #1", "action": "fund", "final_state": "Funding #2"},
                {"initial_state": "Funding #2", "action": "stay", "final_state": "Funding #2"},
                {"initial_state": "Funding #2", "action": "use", "final_state": "Data Collection"},
                {"initial_state": "Data Collection", "action": "use", "final_state": "Data Collection"},
                {"initial_state": "Data Collection", "action": "end", "final_state": "End of Use"},
            ],
        },
        {
            "id": "Space agency",
            "states": {
                "Planning; Design": [],
                "Planning; Manufacturing Satellite #1": [],
                "Acquisition Satellite #2; Manufacturing Satellite #1": [],
                "Acquisition Satellite #2; Operation Satellite #1": [],
                "Operation Satellite #2; Operation Satellite #1": [],
                "Retirement": [],
            },
            "actions": ["stay", "manufacture", "acquire, manufacture", "acquire", "acquire, operate 1", "operate 1",
                        "operate 1,2", "retire"],
            "available_actions": {
                "Planning; Design": ["stay", "manufacture", "acquire, manufacture"],
                "Planning; Manufacturing Satellite #1": ["stay", "acquire", "acquire, operate 1"],
                "Acquisition Satellite #2; Manufacturing Satellite #1": ["stay", "operate 1", "operate 1,2"],
                "Acquisition Satellite #2; Operation Satellite #1": ["operate 1", "operate 1,2"],
                "Operation Satellite #2; Operation Satellite #1": ["operate 1,2", "retire"],
                "Retirement": ["stay"],            # added: terminal state (otherwise ALL actions are available)
            },
            "own_rewards": {
                "Planning; Design": {"stay":-1, "manufacture":-1, "acquire, manufacture":-2},
                "Planning; Manufacturing Satellite #1": {"stay":-1, "acquire":0, "acquire, operate 1":-0.1},
                "Acquisition Satellite #2; Manufacturing Satellite #1": {"stay":-1, "operate 1":-0.1, "operate 1,2":-0.1+-0.2},
                "Acquisition Satellite #2; Operation Satellite #1": {"operate 1":-0.1, "operate 1,2":-0.1-0.2},
                "Operation Satellite #2; Operation Satellite #1": {"operate 1,2":-0.1+-0.2, "retire":-1.2},
            },
            "transitions": [
                {"initial_state": "Planning; Design", "action": "stay", "final_state": "Planning; Design"},
                {"initial_state": "Planning; Design", "action": "manufacture", "final_state": "Planning; Manufacturing Satellite #1"},
                {"initial_state": "Planning; Design", "action": "acquire, manufacture", "final_state": "Acquisition Satellite #2; Manufacturing Satellite #1"},
                {"initial_state": "Planning; Manufacturing Satellite #1", "action": "stay", "final_state": "Planning; Manufacturing Satellite #1"},
                {"initial_state": "Planning; Manufacturing Satellite #1", "action": "acquire", "final_state": "Acquisition Satellite #2; Manufacturing Satellite #1"},
                {"initial_state": "Planning; Manufacturing Satellite #1", "action": "acquire, operate 1", "final_state": "Acquisition Satellite #2; Operation Satellite #1"},
                {"initial_state": "Acquisition Satellite #2; Manufacturing Satellite #1", "action": "stay", "final_state": "Acquisition Satellite #2; Manufacturing Satellite #1"},
                {"initial_state": "Acquisition Satellite #2; Manufacturing Satellite #1", "action": "operate 1", "final_state": "Acquisition Satellite #2; Operation Satellite #1"},
                {"initial_state": "Acquisition Satellite #2; Manufacturing Satellite #1", "action": "operate 1,2", "final_state": "Operation Satellite #2; Operation Satellite #1"},
                {"initial_state": "Acquisition Satellite #2; Operation Satellite #1", "action": "operate 1", "final_state": "Acquisition Satellite #2; Operation Satellite #1"},
                {"initial_state": "Acquisition Satellite #2; Operation Satellite #1", "action": "operate 1,2", "final_state": "Operation Satellite #2; Operation Satellite #1"},
                {"initial_state": "Operation Satellite #2; Operation Satellite #1", "action": "operate 1,2", "final_state": "Operation Satellite #2; Operation Satellite #1"},
                {"initial_state": "Operation Satellite #2; Operation Satellite #1", "action": "retire", "final_state": "Retirement"},
            ],
        },
        {
            "id": "Private company",
            "states": {
                "Low Battery, Low Satellite #2": [],
                "Low Battery, High Satellite #2": [],
                "High Battery, High Satellite #2": [],
                "High Battery, Operation Satellite #2": [],
                "End of Life Cycle": [],
            },
            "actions": ["stay", "develop 2", "develop tech,2", "develop tech", "develop tech, operate 2",
                        "operate 2", "end"],
            "available_actions": {
                "Low Battery, Low Satellite #2": ["stay", "develop 2", "develop tech,2"],
                "Low Battery, High Satellite #2": ["stay", "develop tech", "develop tech, operate 2"],
                "High Battery, High Satellite #2": ["stay", "operate 2", "end"],
                "High Battery, Operation Satellite #2": ["operate 2", "end"],
                "End of Life Cycle": ["stay"],     # added: terminal state (otherwise ALL actions are available)
            },
            "own_rewards": {
                "Low Battery, Low Satellite #2": {"stay":-1, "develop 2":-0.7, "develop tech,2":-0.3-0.7},
                "Low Battery, High Satellite #2": {"stay":-1, "develop tech":-0.3, "develop tech, operate 2":-0.3-0.4},
                "High Battery, High Satellite #2": {"stay":-1, "operate 2":-0.4, "end":-0.5},
                "High Battery, Operation Satellite #2": {"operate 2":-0.4, "end":-0.5},
            },
            "transitions": [
                {"initial_state": "Low Battery, Low Satellite #2", "action": "stay", "final_state": "Low Battery, Low Satellite #2"},
                {"initial_state": "Low Battery, Low Satellite #2", "action": "develop 2", "final_state": "Low Battery, High Satellite #2"},
                {"initial_state": "Low Battery, Low Satellite #2", "action": "develop tech,2", "final_state": "High Battery, High Satellite #2"},
                {"initial_state": "Low Battery, High Satellite #2", "action": "stay", "final_state": "Low Battery, High Satellite #2"},
                {"initial_state": "Low Battery, High Satellite #2", "action": "develop tech", "final_state": "High Battery, High Satellite #2"},
                {"initial_state": "Low Battery, High Satellite #2", "action": "develop tech, operate 2", "final_state": "High Battery, Operation Satellite #2"},
                {"initial_state": "High Battery, High Satellite #2", "action": "stay", "final_state": "High Battery, High Satellite #2"},
                {"initial_state": "High Battery, High Satellite #2", "action": "operate 2", "final_state": "High Battery, Operation Satellite #2"},
                {"initial_state": "High Battery, High Satellite #2", "action": "end", "final_state": "End of Life Cycle"},
                {"initial_state": "High Battery, Operation Satellite #2", "action": "operate 2", "final_state": "High Battery, Operation Satellite #2"},
                {"initial_state": "High Battery, Operation Satellite #2", "action": "end", "final_state": "End of Life Cycle"},
            ],
        }
    ],

    "rules": [
        {
            "name": "system acquisition",
            "when": {"Space agency": {"action": ["acquire", "acquire, manufacture", "acquire, operate 1"]}, "Private company": {"state": "*"}},
            "flow": {"from": "Space agency", "to": "Private company", "amount":3},
        },
        {
            "name": "technology acquisition",
            "when": {"Space agency": {"action": ["manufacture", "acquire, manufacture"]}, "Private company": {"state": "*"}},
            "flow": {"from": "Space agency", "to": "Private company", "amount":1.5},
        },
        {
            "name": "funding #1",
            "when": {"Ministry of Finance": {"state":["Policy Making", "Funding #1"], "action":"fund"}, "Space agency": {"state": "*"}},
            "flow": {"from": "Ministry of Finance", "to": "Space agency", "amount":5},
        },
        {
            "name": "funding #2",
            "when": {"Ministry of Finance": {"state": "Funding #2", "action": "fund"}, "Space agency": {"state": "*"}},
            "flow": {"from": "Ministry of Finance", "to": "Space agency", "amount": 3},
        },
        {
            "name": "service transaction system 1",
            "when": {"Space agency": {"action": ["operate 1", "operate 1,2"]}, "Ministry of Finance": {"action": "use"}},
            "transfers": {"Space agency": 0.25, "Ministry of Finance": -0.3},
        },
        {
            "name": "service transaction system 2",
            "when": {"Space agency": {"action": "operate 1,2"}, "Private company": {"action": ["operate 2", "develop tech, operate 2"]}, "Ministry of Finance": {"action": "use"}},
            "transfers": {"Space agency": 0.15, "Private company": 0.1, "Ministry of Finance": -0.25},
        },
    ]
}
# ------------ End of SoS1 use case ------------ #


# =====================================================================================
# Exploring starts: sequences that start in different stage games
# =====================================================================================
def absorbing_games(game):
    """Stage games that can never be left: every joint action leads back to the same game (p = 1)."""
    return {k for k, g in game.build().items()
            if all(out == [(k, 1.0)] for out in g.transitions.values())}


def _truncate(tr: Trajectory, n: int) -> Trajectory:
    """First n steps of a sequence."""
    return Trajectory(tr.stakeholders, tr.states[:n + 1], tr.games[:n + 1], tr.actions[:n],
                      tr.rewards[:n], tr.rewards_self[:n], tr.rewards_transfer[:n], tr.rules_fired[:n],
                      tr.action_probs[:n] if tr.action_probs else None)


def exploring_start_sequences(game, n_sequences, n_steps, initial_state=None, p_initial=0.2,
                              stop_at_absorbing=True, seed=None):
    """Monte Carlo sequences (random actions) that start in a randomly chosen stage game.

    p_initial         : share of sequences that still start in initial_state (keeps the start well covered);
                        the rest start in a stage game drawn uniformly from all stage games
    stop_at_absorbing : end a sequence after its first step in a stage game that can never be left
                        (e.g. every stakeholder in its end state), so samples are not wasted there
    Returns a list of Trajectory objects (sequences can have different lengths)."""
    rng = np.random.default_rng(seed)
    keys = list(game.build())
    absorbing = absorbing_games(game) if stop_at_absorbing else set()
    sequences = []
    for _ in range(n_sequences):
        if initial_state is not None and rng.random() < p_initial:
            start = tuple(initial_state)
        else:
            start = keys[int(rng.integers(len(keys)))]
        tr = game.simulate(n_steps, start, policy="random", rng=rng)
        if absorbing:
            first = next((t for t, js in enumerate(tr.states[:-1]) if js in absorbing), None)
            if first is not None:
                tr = _truncate(tr, first + 1)            # keep one step inside the absorbing game
        sequences.append(tr)
    return sequences


# =====================================================================================
# Nash Q-learning from Monte Carlo sequences
# =====================================================================================
def nash_q_learning(game, trajectories, beta=0.9, count_per="state", Q_init=None,
                    visits_init=None, record_every=1, monitor_state=None, track_states=None,
                    verbose=True):
    """Learn Q values from simulated sequences (Trajectory objects from game.monte_carlo).

    count_per : "state"        -> n = visits of the stage game (alpha shared by all its joint actions)
                "state_action" -> n = visits of the (stage game, joint action) pair (Hu & Wellman)
    Q_init / visits_init : continue learning from an earlier run (e.g. more sequences later)
    monitor_state        : stage game whose Nash value is recorded during learning (learning curve)
    track_states         : stage games whose full Q tables are recorded during learning (for plots);
                           default: [monitor_state]
    Returns a dict with Q tables, visit counts and the learning history.
    """
    games = game.build()
    Q = {k: np.zeros_like(g.R) for k, g in games.items()} if Q_init is None else \
        {k: np.array(q, dtype=float) for k, q in Q_init.items()}
    visits = {} if visits_init is None else dict(visits_init)        # update counts
    cache = {}                       # stage game -> (Nash values, probabilities); cleared when its Q changes
    n_solves = 0

    def nash(k):
        nonlocal n_solves
        if k not in cache:
            vals, probs = NashEquilibriumMixedStrategyProbs(list(Q[k]))
            cache[k] = (np.asarray(vals, dtype=float), probs)
            n_solves += 1
        return cache[k]

    history = {"sequence": [], "updates": [], "max_change": [], "pairs_visited": [], "monitor_value": []}
    n_updates, t0 = 0, time.time()
    seen_pairs = set()                # (stage game, joint action) pairs updated at least once
    pair_updates = {}                 # (stage game, joint action) -> number of updates
    if track_states is None:
        track_states = [monitor_state] if monitor_state is not None else []
    track_states = [tuple(k) for k in track_states]
    history["tracked_Q"] = {k: [] for k in track_states}       # Q-table snapshots per record point
    history["tracked_nash"] = {k: [] for k in track_states}    # Nash value snapshots per record point
    for r, tr in enumerate(trajectories):
        max_change = 0.0
        for t, joint_action in enumerate(tr.actions):
            s, s_next = tr.states[t], tr.states[t + 1]
            g = games[s]
            a = tuple(g.actions[k].index(act) for k, act in enumerate(joint_action))   # action indices
            key = s if count_per == "state" else (s, a)
            n = visits.get(key, 0)
            # alpha = 1.0 / (n + 1)
            alpha = 1

            NashQ_next = nash(s_next)[0]                                 # Nash value of next stage game
            new = (1 - alpha) * Q[s][a] + alpha * (g.R[a] + beta * NashQ_next)
            max_change = max(max_change, float(np.abs(new - Q[s][a]).max()))
            Q[s][a] = new
            visits[key] = n + 1
            seen_pairs.add((s, a))
            pair_updates[(s, a)] = pair_updates.get((s, a), 0) + 1
            cache.pop(s, None)                                           # Q of s changed
            n_updates += 1

        if (r + 1) % record_every == 0 or r == len(trajectories) - 1:
            history["sequence"].append(r + 1)
            history["updates"].append(n_updates)
            history["max_change"].append(max_change)
            history["pairs_visited"].append(len(seen_pairs))
            if monitor_state is not None:
                history["monitor_value"].append(nash(tuple(monitor_state))[0].tolist())
            for k in track_states:
                history["tracked_Q"][k].append(Q[k].copy())
                history["tracked_nash"][k].append(nash(k)[0].copy())
        if verbose and ((r + 1) % max(1, len(trajectories) // 10) == 0):
            print(f"  sequence {r + 1:>6}/{len(trajectories)}  updates {n_updates:>8}  "
                  f"solver calls {n_solves:>7}  max change {max_change:8.4f}  {time.time() - t0:6.1f} s")

    return {"Q": Q, "visits": visits, "pairs": seen_pairs, "pair_updates": pair_updates,
            "history": history, "n_updates": n_updates,
            "n_solves": n_solves, "count_per": count_per}


def coverage(game, learned):
    """How much of the game the sequences explored: stage games and (game, joint action) pairs updated."""
    games = game.build()
    n_pairs = sum(int(np.prod(g.R.shape[:-1])) for g in games.values())
    pairs = learned["pairs"]
    return {"stage_games": (len({s for s, _ in pairs}), len(games)), "joint_actions": (len(pairs), n_pairs)}


def save_learning(path, game, learned):
    data = {
        "count_per": learned["count_per"],
        "n_updates": learned["n_updates"],
        "Q": {f"Q_game_{game.index_name(k)}": q.tolist() for k, q in learned["Q"].items()},
        "visits": {(f"Q_game_{game.index_name(k)}" if learned["count_per"] == "state"
                    else f"Q_game_{game.index_name(k[0])}{list(k[1])}"): v
                   for k, v in learned["visits"].items()},
        "history": {k: v for k, v in learned["history"].items() if not k.startswith("tracked")},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


# =====================================================================================
# Plots: evolution of Q values during learning
# =====================================================================================
PLOT_STYLE = {                       # validated categorical palette (light mode), fixed order
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "nash": "#52514e",               # Nash value of the stage game: neutral ink, dashed
    "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df",
}


def _style_axis(ax):
    ax.set_facecolor(PLOT_STYLE["surface"])
    ax.grid(True, color=PLOT_STYLE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PLOT_STYLE["grid"])
    ax.tick_params(colors=PLOT_STYLE["text2"], labelsize=9)


def _end_labels(ax, x_end, y_ends, labels, colors):
    """Direct labels at the right end of the lines, spread apart inside the panel so they don't overlap."""
    lo, hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    gap = 0.07 * (hi - lo)
    order = np.argsort(y_ends)
    pos = np.asarray(y_ends, dtype=float)[order].copy()
    for i in range(1, len(pos)):                      # push up to keep the gap
        pos[i] = max(pos[i], pos[i - 1] + gap)
    if pos[-1] > hi:                                  # keep inside the panel: shift down from the top
        pos -= pos[-1] - hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    y_text = np.empty_like(pos)
    y_text[order] = pos
    dx = 0.015 * (x_hi - x_lo)
    for y0, y1, label, color in zip(y_ends, y_text, labels, colors):
        moved = abs(y1 - y0) > 1e-9 * max(1.0, abs(hi - lo))
        ax.annotate(label, xy=(x_end, y0), xytext=(x_end + dx, y1), textcoords="data",
                    va="center", fontsize=9, color=PLOT_STYLE["text"], annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8) if moved else None)


def plot_q_evolution(game, learned, state, joint_actions=None, top=4, path=None, show=False):
    """Q values of one stage game over the learning sequences: one panel per stakeholder,
    one line per joint action (default: the `top` most updated ones), plus the Nash value."""
    import matplotlib.pyplot as plt
    state = tuple(state)
    games = game.build()
    g = games[state]
    hist = learned["history"]
    x = np.array(hist["sequence"])
    snaps = np.stack(hist["tracked_Q"][state])                  # (records, a_1, ..., a_n, agents)
    nash_vals = np.stack(hist["tracked_nash"][state])          # (records, agents)

    if joint_actions is None:                                   # most updated joint actions
        counts = {a: c for (s, a), c in learned["pair_updates"].items() if s == state}
        joint_actions = [a for a, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]
    else:                                                       # given as action names
        joint_actions = [tuple(g.actions[k].index(n) for k, n in enumerate(ja)) if isinstance(ja[0], str)
                         else tuple(ja) for ja in joint_actions]
    joint_actions = joint_actions[:len(PLOT_STYLE["series"])]
    names = [", ".join(g.actions[k][j] for k, j in enumerate(a)) for a in joint_actions]
    codes = [f"a{i + 1}" for i in range(len(joint_actions))]
    colors = PLOT_STYLE["series"][:len(joint_actions)]
    n_agents = len(game.ids)

    fig, axes = plt.subplots(1, n_agents, figsize=(4.6 * n_agents, 4.2), sharex=True, squeeze=False)
    fig.patch.set_facecolor(PLOT_STYLE["surface"])
    for i, (ax, sid) in enumerate(zip(axes[0], game.ids)):
        _style_axis(ax)
        ends = []
        for a, c in zip(joint_actions, colors):
            y = snaps[(slice(None),) + a + (i,)]
            ax.plot(x, y, color=c, linewidth=2)
            ends.append(y[-1])
        ax.plot(x, nash_vals[:, i], color=PLOT_STYLE["nash"], linewidth=2, linestyle=(0, (4, 3)))
        ends.append(nash_vals[-1, i])
        _end_labels(ax, x[-1], np.array(ends), codes + ["Nash"], colors + [PLOT_STYLE["nash"]])
        ax.set_title(sid, fontsize=11, color=PLOT_STYLE["text"], loc="left")
        ax.set_xlabel("Sequences used for learning", fontsize=9, color=PLOT_STYLE["text2"])
        if i == 0:
            ax.set_ylabel("Q value", fontsize=9, color=PLOT_STYLE["text2"])
        ax.margins(x=0.02)

    handles = [plt.Line2D([], [], color=c, linewidth=2) for c in colors] + \
              [plt.Line2D([], [], color=PLOT_STYLE["nash"], linewidth=2, linestyle=(0, (4, 3)))]
    labels = [f"{code}: ({name})" for code, name in zip(codes, names)] + ["Nash value of the stage game"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=min(3, len(labels)),
               frameon=False, fontsize=9, labelcolor=PLOT_STYLE["text"])
    head = " | ".join(state)
    fig.suptitle(f"Q values of stage game {g.name} during learning   [{head}]",
                 x=0.01, ha="left", fontsize=12, color=PLOT_STYLE["text"])
    fig.tight_layout(rect=(0, 0, 0.97, 0.84))
    if path:
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_learning_diagnostics(game, learned, path=None, show=False):
    """How learning progresses: largest Q change per sequence, and how much of the game was explored."""
    import matplotlib.pyplot as plt
    hist = learned["history"]
    x = np.array(hist["sequence"])
    n_pairs = sum(int(np.prod(g.R.shape[:-1])) for g in game.build().values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor(PLOT_STYLE["surface"])
    for ax in (ax1, ax2):
        _style_axis(ax)
        ax.set_xlabel("Sequences used for learning", fontsize=9, color=PLOT_STYLE["text2"])

    ax1.plot(x, np.maximum(hist["max_change"], 1e-12), color=PLOT_STYLE["series"][0], linewidth=2)
    ax1.set_yscale("log")
    ax1.set_title("Largest Q change in the sequence", fontsize=11, color=PLOT_STYLE["text"], loc="left")
    ax1.set_ylabel("max |change of Q|", fontsize=9, color=PLOT_STYLE["text2"])

    ax2.plot(x, hist["pairs_visited"], color=PLOT_STYLE["series"][0], linewidth=2)
    ax2.axhline(n_pairs, color=PLOT_STYLE["text2"], linewidth=1, linestyle=(0, (4, 3)))
    ax2.annotate(f"all {n_pairs} pairs", xy=(x[0], n_pairs), xytext=(0, 4), textcoords="offset points",
                 fontsize=9, color=PLOT_STYLE["text2"])
    ax2.set_ylim(0, n_pairs * 1.12)
    ax2.set_title("(Stage game, joint action) pairs updated", fontsize=11, color=PLOT_STYLE["text"], loc="left")
    ax2.set_ylabel("pairs updated at least once", fontsize=9, color=PLOT_STYLE["text2"])

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# =====================================================================================
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)

    # ---- settings
    BETA = 0.99
    INITIAL_STATE = ("Policy Making", "Planning; Design", "Low Battery, Low Satellite #2")
    N_SEQUENCES = 20000      # number of Monte Carlo sequences used for learning
    N_STEPS = 8           # time steps per sequence
    SEED = 1               # reproducibility
    COUNT_PER = "state"    # alpha = 1/(n+1), n = visits of the stage game ("state_action": per joint action)
    RECORD_EVERY = 1000       # record Q values every N sequences (for the plots)
    TRACK_STATES = [INITIAL_STATE,                                                    # stage games to plot
                    ("Funding #1", "Planning; Manufacturing Satellite #1", "Low Battery, High Satellite #2")]
    SHOW_PLOTS = True      # open the plot windows (they are also saved as PNG)
    EXPLORING_STARTS = True    # start sequences in randomly chosen stage games
    P_INITIAL = 0.2            # share of sequences that still start in INITIAL_STATE
    STOP_AT_ABSORBING = False   # end a sequence once it reaches a stage game that can never be left

    game = MarkovGame.from_spec(SoS1)
    games = game.build()
    print(game.summary(), "\n")

    # ---- 1. generate sequences of stage games with the Monte Carlo simulation (random actions)
    if EXPLORING_STARTS:      # start in randomly chosen stage games (P_INITIAL share still from INITIAL_STATE)
        sequences = exploring_start_sequences(game, N_SEQUENCES, N_STEPS, initial_state=INITIAL_STATE,
                                              p_initial=P_INITIAL, stop_at_absorbing=STOP_AT_ABSORBING,
                                              seed=SEED)
    else:                     # every sequence starts in INITIAL_STATE
        sequences = game.monte_carlo(n_runs=N_SEQUENCES, n_steps=N_STEPS, initial_state=INITIAL_STATE,
                                     policy="random", seed=SEED).trajectories
    starts = {tr.games[0] for tr in sequences}
    print(f"Generated {len(sequences)} sequences, {sum(len(tr.actions) for tr in sequences)} steps in total, "
          f"starting in {len(starts)} different stage games; example sequence of stage games:")
    print("  " + " -> ".join(sequences[0].games), "\n")

    # ---- 2. learn the Q values from these sequences
    print("Nash Q-learning:")
    learned = nash_q_learning(game, sequences, beta=BETA, count_per=COUNT_PER,
                              monitor_state=INITIAL_STATE, track_states=TRACK_STATES,
                              record_every=RECORD_EVERY)
    Q = learned["Q"]
    print(f"{learned['n_updates']} updates, {learned['n_solves']} solver calls\n")

    cov = coverage(game, learned)
    print(f"Explored: {cov['stage_games'][0]}/{cov['stage_games'][1]} stage games, "
          f"{cov['joint_actions'][0]}/{cov['joint_actions'][1]} (stage game, joint action) pairs")
    visits = learned["visits"]
    if COUNT_PER == "state":
        top = sorted(visits.items(), key=lambda kv: -kv[1])[:5]
        print("Most visited stage games: " +
              ", ".join(f"{games[k].name} ({v})" for k, v in top))
    print()

    # learned Q tables in the generated-script format, e.g. Q_named["Q_game_0_0_0"]
    Q_named = {f"Q_game_{game.index_name(k)}": list(q) for k, q in Q.items()}
    # print("Q_game_0_0_0 (initial stage game):")
    # print(np.asarray(Q_named["Q_game_0_0_0"]), "\n")

    # ---- 3. best policy from the learned Q values (same solver) and its sequence of stage games
    res = game.policy_from_q_tables(Q_named)
    best = game.best_policy_sequence(res, n_steps=N_STEPS, initial_state=INITIAL_STATE)
    print("Best-policy sequence from the learned Q values:")
    print(best.table(), "\n")

    save_learning("sos1_nash_q_learning.json", game, learned)
    print("Wrote sos1_nash_q_learning.json (Q tables, visit counts, learning history)")

    # ---- 4. plots: evolution of Q values during learning
    for k in TRACK_STATES:
        name = games[tuple(k)].name
        plot_q_evolution(game, learned, k, top=4, path=f"q_evolution_{name}.png")
        # or choose the joint actions yourself, e.g.
        # plot_q_evolution(game, learned, k, joint_actions=[("fund", "manufacture", "develop 2")])
        print(f"Wrote q_evolution_{name}.png")
    plot_learning_diagnostics(game, learned, path="learning_diagnostics.png")
    print("Wrote learning_diagnostics.png")
    if SHOW_PLOTS:
        import matplotlib.pyplot as plt
        plt.show()