"""
Monte Carlo sampling of the SoS1 design variables (no optimization).

    design variables : selected own-reward values and transfer-rule amounts: continuous (Real), integer
                       (Integer) or one of a list of levels (Choice), each drawn at random within its bounds
    outputs          : final cumulative reward and minimum cumulative reward of each stakeholder per design;
                       feasibility against MIN_REWARD_BOUND, the Pareto front of the feasible designs and the
                       best feasible final reward found so far, i.e. the same outputs as the Bayesian
                       optimization and NSGA-II scripts, so the three can be compared at equal numbers of
                       evaluations (Monte Carlo = the random-search baseline).

Every design point is evaluated with evaluate_sos1 (Nash Q-value iteration + best-policy sequence).

Needs, in the same folder: sos1_evaluate.py and SoS_Markov_game_with_transitions.py
Needs pymoo only for the Real / Integer / Choice declarations (same as the NSGA-II script).
"""
import csv
import multiprocessing
import time

import numpy as np

from pymoo.core.variable import Real, Integer, Choice

from function_for_optimization import evaluate_sos1

# =====================================================================================
# 1. Settings
# =====================================================================================
STAKEHOLDERS = ["Ministry of Finance", "Space agency", "Private company"]

# Design variables: (kind, where, variable)  -- same format as in the NSGA-II script
#   kind "rule": where = (rule name,)                       for "flow" rules (amount moved)
#                where = (rule name, stakeholder)           for "transfers" rules
#   kind "own" : where = (stakeholder, state, action)       own reward of that action in that state
#   variable   : Real(bounds=(lower, upper))                continuous, uniform between the bounds
#                Integer(bounds=(lower, upper))             whole numbers lower, ..., upper, each equally likely
#                Choice(options=[v1, v2, ...])              one of a list of levels, each equally likely
DESIGN_VARIABLES = [
    ("rule", ("funding #1",),                                   Integer(bounds=(3, 8))),
    ("rule", ("system acquisition",),                           Real(bounds=(1.0, 5.0))),
    ("rule", ("technology acquisition",),                       Real(bounds=(0.5, 3.0))),
    ("rule", ("service transaction system 1", "Space agency"),  Choice(options=[0.0, 0.25, 0.5, 0.75, 1.0])),
    ("own",  ("Ministry of Finance", "Data Collection", "use"), Real(bounds=(1.0, 4.0))),
    ("own",  ("Private company", "High Battery, Operation Satellite #2", "operate 2"), Real(bounds=(-1.0, 0.0))),
]

# Constraints: minimum cumulative reward over the time steps must stay >= these bounds (feasibility only)
MIN_REWARD_BOUND = {"Ministry of Finance": -12.0, "Space agency": -2.0, "Private company": -2.0}

# Evaluation settings (passed to evaluate_sos1)
EVAL_SETTINGS = dict(beta=0.99, n_iter=10, n_steps=8)

# Monte Carlo settings: N_SAMPLES evaluations (about 30 s each with n_iter=50)
N_SAMPLES = 8
SAMPLING = "random"   # "random": every design independent and uniform (plain Monte Carlo)
                      # "lhs"   : Latin hypercube, spreads the N_SAMPLES designs evenly over each variable's range
SEED = 1
N_PROCESSES = 1       # > 1: evaluate the designs in parallel worker processes (e.g. number of CPU cores)

RESULTS_CSV = "sos1_mc_results.csv"         # all evaluated points; None = don't write
SEQUENCES_CSV = "sos1_mc_sequences.csv"     # step-by-step best-policy sequences; None = don't write
SEQUENCES_FOR = "pareto"                    # "pareto": only Pareto-optimal designs, "all": every evaluated design
PLOT_PATH = "sos1_mc_pareto.png"            # Pareto plot (also saved as PDF); None = don't save
BEST_PLOT_PATH = "sos1_mc_best_so_far.png"  # best feasible final reward so far vs evaluations (also PDF); None = don't save
SHOW_PLOT = True                            # open the plot windows
SHOW_INFEASIBLE = True                      # also draw infeasible points (faint), for context


# =====================================================================================
# 2. Sampling the design variables and turning them into the inputs of evaluate_sos1
# =====================================================================================
NAMES = [" / ".join(where) for _, where, _ in DESIGN_VARIABLES]


def sample_designs(n, method="random", seed=None):
    """n designs (list of value lists, same order as DESIGN_VARIABLES).
    A number u in [0, 1) is drawn per design and variable (independently for "random", one per stratum
    for "lhs") and mapped to the variable: Real -> lower + u (upper - lower); Integer and Choice -> the
    level with index floor(u x number of levels), so every level is equally likely."""
    rng = np.random.default_rng(seed)
    n_var = len(DESIGN_VARIABLES)
    if method == "random":
        U = rng.random((n, n_var))
    elif method == "lhs":
        U = (np.argsort(rng.random((n, n_var)), axis=0) + rng.random((n, n_var))) / n
    else:
        raise ValueError("SAMPLING must be 'random' or 'lhs'")
    designs = []
    for u in U:
        values = []
        for uj, (_, _, var) in zip(u, DESIGN_VARIABLES):
            if isinstance(var, Real):
                lo, hi = var.bounds
                values.append(float(lo + uj * (hi - lo)))
            elif isinstance(var, Integer):
                lo, hi = var.bounds
                values.append(float(min(int(lo + np.floor(uj * (hi - lo + 1))), hi)))
            elif isinstance(var, Choice):
                values.append(float(var.options[min(int(np.floor(uj * len(var.options))), len(var.options) - 1)]))
            else:
                raise TypeError(f"use Real, Integer or Choice for design variables, not {type(var).__name__}")
        designs.append(values)
    return designs


def to_inputs(values):
    """values (actual design values, same order as DESIGN_VARIABLES)
    -> own_rewards, rule_amounts dictionaries for evaluate_sos1."""
    own_rewards, rule_amounts = {}, {}
    for value, (kind, where, _) in zip(values, DESIGN_VARIABLES):
        value = float(value)
        if kind == "own":
            sid, state, action = where
            own_rewards.setdefault(sid, {}).setdefault(state, {})[action] = value
        elif kind == "rule" and len(where) == 1:
            rule_amounts[where[0]] = value
        elif kind == "rule":
            rule_amounts.setdefault(where[0], {})[where[1]] = value
        else:
            raise ValueError(f"unknown design variable kind '{kind}'")
    return own_rewards, rule_amounts



# =====================================================================================
# 3. Evaluation of one design
# =====================================================================================
_CACHE = {}     # design values -> (final, minimum, sequence); one cache per process


def evaluate_design(values):
    """One SoS1 evaluation (cached, so a repeated design is not recomputed)."""
    key = tuple(np.round(values, 12))
    if key not in _CACHE:
        own_rewards, rule_amounts = to_inputs(values)
        final, minimum, details = evaluate_sos1(own_rewards=own_rewards, rule_amounts=rule_amounts,
                                                return_details=True, **EVAL_SETTINGS)
        traj = details["trajectory"]
        sequence = {"games": list(traj.games), "states": [list(s) for s in traj.states],
                    "actions": [list(a) for a in traj.actions], "rewards": traj.rewards.tolist(),
                    "converged": bool(details["converged"])}
        _CACHE[key] = (final, minimum, sequence)
    return _CACHE[key]



# =====================================================================================
# 4. Evaluating all sampled designs (in sample order, optionally in parallel)
# =====================================================================================
def evaluate_all(designs, n_processes=1):
    """-> list of (final, minimum, sequence), in the same order as designs (= evaluation order)."""
    results, t0 = [], time.time()

    def report(i, final, minimum):
        ok = all(minimum[s] >= MIN_REWARD_BOUND[s] for s in STAKEHOLDERS)
        print(f"  evaluation {i + 1:>4}/{len(designs)}  final " +
              ", ".join(f"{final[s]:7.2f}" for s in STAKEHOLDERS) +
              f"  {'feasible  ' if ok else 'infeasible'}  {(time.time() - t0) / 60:6.1f} min")

    if n_processes > 1:
        with multiprocessing.Pool(n_processes) as pool:
            for i, res in enumerate(pool.imap(evaluate_design, designs)):     # imap keeps the order
                results.append(res)
                report(i, res[0], res[1])
    else:
        for i, values in enumerate(designs):
            results.append(evaluate_design(values))
            report(i, results[-1][0], results[-1][1])
    return results


# =====================================================================================
# 5. Pareto front of the feasible evaluated points
# =====================================================================================
def pareto_mask(F):
    """Non-dominated rows of F (all objectives minimized)."""
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if keep[i]:
            dominated = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
            keep[dominated] = False
    return keep


# =====================================================================================
# 6. Plot: Pareto points and all feasible points, pairwise for the three objectives
# =====================================================================================
PLOT_STYLE = {"pareto": "#2a78d6", "feasible": "#8a8984", "infeasible": "#c9c8c3",
              "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df"}


def plot_pareto(final, feasible, pareto_idx, path=None, show=True, show_infeasible=True):
    """Final cumulative rewards of the evaluated designs, one panel per pair of stakeholders.
    final: (n, 3) final rewards; feasible: (n,) bool; pareto_idx: indices of the Pareto-optimal designs."""
    import matplotlib.pyplot as plt
    pairs = [(0, 1), (0, 2), (1, 2)]
    is_pareto = np.zeros(len(final), dtype=bool)
    is_pareto[pareto_idx] = True
    feas_only = feasible & ~is_pareto

    fig, axes = plt.subplots(1, len(pairs), figsize=(14, 4.8))
    fig.patch.set_facecolor(PLOT_STYLE["surface"])
    for ax, (i, j) in zip(axes, pairs):
        ax.set_facecolor(PLOT_STYLE["surface"])
        ax.grid(True, color=PLOT_STYLE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PLOT_STYLE["grid"])
        ax.tick_params(colors=PLOT_STYLE["text2"], labelsize=9)

        if show_infeasible and (~feasible).any():
            ax.scatter(final[~feasible, i], final[~feasible, j], s=40, marker="x", linewidths=1.2,
                       color=PLOT_STYLE["infeasible"], zorder=2)
        ax.scatter(final[feas_only, i], final[feas_only, j], s=55, color=PLOT_STYLE["feasible"],
                   edgecolors=PLOT_STYLE["surface"], linewidths=1.5, zorder=3)
        ax.scatter(final[is_pareto, i], final[is_pareto, j], s=90, color=PLOT_STYLE["pareto"],
                   edgecolors=PLOT_STYLE["surface"], linewidths=1.5, zorder=4)
        # point numbers (as in the printout and CSV); points almost on top of each other share one label
        xr = np.ptp(final[:, i]) or 1.0
        yr = np.ptp(final[:, j]) or 1.0
        groups = []
        for k in sorted(pareto_idx, key=lambda k: (final[k, i], final[k, j])):
            for g in groups:
                if abs(final[k, i] - final[g[0], i]) < 0.04 * xr and abs(final[k, j] - final[g[0], j]) < 0.04 * yr:
                    g.append(k)
                    break
            else:
                groups.append([k])
        for g in groups:
            ax.annotate(", ".join(str(k) for k in sorted(g)), (final[g[0], i], final[g[0], j]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color=PLOT_STYLE["text"])
        ax.set_xlabel(f"Final reward: {STAKEHOLDERS[i]}", fontsize=10, color=PLOT_STYLE["text2"])
        ax.set_ylabel(f"Final reward: {STAKEHOLDERS[j]}", fontsize=10, color=PLOT_STYLE["text2"])

    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=9, color=PLOT_STYLE["pareto"]),
               plt.Line2D([], [], linestyle="", marker="o", markersize=7, color=PLOT_STYLE["feasible"])]
    labels = [f"Pareto-optimal ({len(pareto_idx)}), labelled with point number",
              f"Feasible, not Pareto-optimal ({int(feas_only.sum())})"]
    if show_infeasible and (~feasible).any():
        handles.append(plt.Line2D([], [], linestyle="", marker="x", markersize=7, markeredgewidth=1.2,
                                  color=PLOT_STYLE["infeasible"]))
        labels.append(f"Infeasible ({int((~feasible).sum())})")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=PLOT_STYLE["text"])
    fig.suptitle("Monte Carlo sampling of SoS1: final cumulative reward of each stakeholder (higher is better)",
                 x=0.01, ha="left", fontsize=13, color=PLOT_STYLE["text"])
    fig.text(0.01, 0.015, "Pareto-optimal over all three stakeholders together, so in a two-stakeholder panel "
                          "a Pareto point can appear dominated.", fontsize=8.5, color=PLOT_STYLE["text2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.88), w_pad=3)
    if path:
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(path.rsplit(".", 1)[0] + ".pdf", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def best_so_far(final, feasible):
    """Best (maximum) final cumulative reward of each stakeholder over the FEASIBLE evaluations up to and
    including each evaluation, in evaluation order. If an evaluation is no better (or is infeasible),
    the value stays the same as after the previous evaluation. NaN until the first feasible evaluation.
    final: (n, k) final rewards in evaluation order; feasible: (n,) bool -> (n, k)."""
    values = np.where(np.asarray(feasible)[:, None], final, -np.inf)
    best = np.maximum.accumulate(values, axis=0)
    best[np.isneginf(best)] = np.nan
    return best


def plot_best_so_far(final, feasible, title, path=None, show=True, boundaries=(), boundary_note=None,
                     show_evaluations=True):
    """One panel per stakeholder: best feasible final reward found so far vs number of function evaluations.
    boundaries: x positions of dashed vertical lines (e.g. end of the initial design, generation boundaries)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    best = best_so_far(final, feasible)
    n = len(final)
    x = np.arange(1, n + 1)                               # evaluation number (= point number + 1)

    fig, axes = plt.subplots(1, len(STAKEHOLDERS), figsize=(14, 4.6), sharex=True)
    fig.patch.set_facecolor(PLOT_STYLE["surface"])
    for k, (ax, sid) in enumerate(zip(axes, STAKEHOLDERS)):
        ax.set_facecolor(PLOT_STYLE["surface"])
        ax.grid(True, color=PLOT_STYLE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PLOT_STYLE["grid"])
        ax.tick_params(colors=PLOT_STYLE["text2"], labelsize=9)
        for b in boundaries:
            ax.axvline(b, color=PLOT_STYLE["feasible"], linewidth=1, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)

        if show_evaluations:                              # each feasible evaluation, for context
            ax.scatter(x[feasible], final[feasible, k], s=16, color=PLOT_STYLE["feasible"], alpha=0.5,
                       linewidths=0, zorder=2)
        # running maximum as a step line, held flat until the next improvement (and to the last evaluation)
        ax.plot(np.r_[x, n + 0.5], np.r_[best[:, k], best[-1, k]], drawstyle="steps-post",
                color=PLOT_STYLE["pareto"], linewidth=2, zorder=3)
        b = np.where(np.isnan(best[:, k]), -np.inf, best[:, k])
        new = np.r_[b[0] > -np.inf, b[1:] > b[:-1]]        # evaluations that set a new best
        ax.scatter(x[new], best[new, k], s=42, color=PLOT_STYLE["pareto"], edgecolors=PLOT_STYLE["surface"],
                   linewidths=1.5, zorder=4)
        if new.any():
            last = int(np.where(new)[0][-1])
            ax.annotate(f"{best[-1, k]:.2f} (evaluation {last + 1})", (x[last], best[last, k]),
                        xytext=(0, 9), textcoords="offset points", ha="center" if last < 0.8 * n else "right",
                        fontsize=8.5, color=PLOT_STYLE["text"])
        ax.set_title(sid, loc="left", fontsize=11, color=PLOT_STYLE["text"])
        ax.set_xlabel("Function evaluation", fontsize=10, color=PLOT_STYLE["text2"])
        ax.set_xlim(0.5, n + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.12)
    axes[0].set_ylabel("Best feasible final reward so far", fontsize=10, color=PLOT_STYLE["text2"])

    handles = [plt.Line2D([], [], color=PLOT_STYLE["pareto"], linewidth=2),
               plt.Line2D([], [], linestyle="", marker="o", markersize=7, color=PLOT_STYLE["pareto"])]
    labels = ["Best feasible final reward so far", "New best"]
    if show_evaluations:
        handles.append(plt.Line2D([], [], linestyle="", marker="o", markersize=5, color=PLOT_STYLE["feasible"],
                                  alpha=0.5))
        labels.append("Feasible evaluation")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=PLOT_STYLE["text"])
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=PLOT_STYLE["text"])
    note = ("Each stakeholder's maximum is taken separately. Infeasible evaluations do not count; "
            "a line starts at the first feasible evaluation.")
    if boundary_note:
        note += " " + boundary_note
    fig.text(0.01, 0.015, note, fontsize=8.5, color=PLOT_STYLE["text2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.88), w_pad=3)
    if path:
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(path.rsplit(".", 1)[0] + ".pdf", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# =====================================================================================
# 7. Monte Carlo run
# =====================================================================================
if __name__ == "__main__":
    designs = sample_designs(N_SAMPLES, method=SAMPLING, seed=SEED)
    print(f"Monte Carlo ({SAMPLING} sampling): {len(DESIGN_VARIABLES)} design variables, {N_SAMPLES} designs, "
          f"{N_PROCESSES} process(es)")
    t0 = time.time()
    evaluated = evaluate_all(designs, n_processes=N_PROCESSES)
    print(f"\nDone: {len(evaluated)} SoS1 evaluations in {(time.time() - t0) / 60:.1f} min")

    # ---- results over ALL evaluated designs (in evaluation order)
    V = np.array(designs)                                                   # (n, n_var) design values
    final = np.array([[f[s] for s in STAKEHOLDERS] for f, _, _ in evaluated])     # (n, 3) final rewards
    minimum = np.array([[m[s] for s in STAKEHOLDERS] for _, m, _ in evaluated])   # (n, 3) minimum rewards
    feasible = np.all(minimum >= np.array([MIN_REWARD_BOUND[s] for s in STAKEHOLDERS]) - 1e-9, axis=1)
    sequences = [sq for _, _, sq in evaluated]
    print(f"{feasible.sum()} of {len(V)} evaluated points satisfy all constraints")
    best = best_so_far(final, feasible)            # (n, 3) best feasible final reward up to each evaluation
    if feasible.any():
        print("Best feasible final reward of each stakeholder (first reached at):")
        for k, s in enumerate(STAKEHOLDERS):
            i = int(np.nanargmax(np.where(feasible, final[:, k], np.nan)))
            print(f"  {s:<20} {final[i, k]:8.2f}   evaluation {i + 1} (point {i})")

    idx = np.array([], dtype=int)
    if feasible.any():
        idx = np.where(feasible)[0][pareto_mask(-final[feasible])]
        idx = idx[np.argsort(-final[idx, 0])]
        print(f"\nPareto-optimal designs ({len(idx)}): no other feasible design is better for every stakeholder")
        for i in idx:
            print(f"  point {i:>3}: final reward " +
                  ", ".join(f"{s} {final[i, k]:.2f}" for k, s in enumerate(STAKEHOLDERS)) +
                  "  | min " + ", ".join(f"{minimum[i, k]:.2f}" for k in range(len(STAKEHOLDERS))))
            print("             " + "; ".join(f"{n} = {v:.3g}" for n, v in zip(NAMES, V[i])))
            sq = sequences[i]
            print("             games  : " + " -> ".join(sq["games"]))
            print("             actions: " + " | ".join(f"t{t} ({', '.join(a)})" for t, a in enumerate(sq["actions"])))

    if RESULTS_CSV:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["point", "evaluation"] + NAMES + [f"final | {s}" for s in STAKEHOLDERS]
                       + [f"min | {s}" for s in STAKEHOLDERS] + ["feasible", "pareto"]
                       + [f"best so far | {s}" for s in STAKEHOLDERS] + ["game sequence"])
            for i in range(len(V)):
                w.writerow([i, i + 1] + list(V[i]) + list(final[i]) + list(minimum[i])
                           + [bool(feasible[i]), bool(i in idx)]
                           + ["" if np.isnan(v) else v for v in best[i]] + [" -> ".join(sequences[i]["games"])])
        print(f"\nWrote {RESULTS_CSV}")

    # ---- distinct sequences of stage games among the evaluated designs
    groups = {}
    for i in range(len(V)):
        groups.setdefault(tuple(sequences[i]["games"]), []).append(i)
    print(f"\n{len(groups)} distinct best-policy sequences of stage games among {len(V)} evaluated designs:")
    for seq, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n_feas = int(feasible[pts].sum())
        n_par = len(set(pts) & set(idx.tolist()))
        print(f"  {len(pts):>3} designs ({n_feas} feasible, {n_par} Pareto): " + " -> ".join(seq))

    if SEQUENCES_CSV:
        chosen = idx if SEQUENCES_FOR == "pareto" else np.arange(len(V))
        with open(SEQUENCES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["point", "pareto", "step", "game"] + [f"state | {s}" for s in STAKEHOLDERS]
                       + [f"action | {s}" for s in STAKEHOLDERS] + [f"reward | {s}" for s in STAKEHOLDERS]
                       + [f"cum_reward | {s}" for s in STAKEHOLDERS] + ["next game"])
            for i in chosen:
                sq = sequences[i]
                rewards = np.array(sq["rewards"])
                cum = np.cumsum(rewards, axis=0)
                for t, a in enumerate(sq["actions"]):
                    w.writerow([i, bool(i in idx), t, sq["games"][t]] + list(sq["states"][t]) + list(a)
                               + list(rewards[t]) + list(cum[t]) + [sq["games"][t + 1]])
        print(f"Wrote {SEQUENCES_CSV} (step-by-step sequences of the "
              f"{'Pareto-optimal' if SEQUENCES_FOR == 'pareto' else 'evaluated'} designs)")

    # ---- plots: Pareto points and all feasible points; best feasible final reward so far
    if feasible.any():
        plot_pareto(final, feasible, idx, path=PLOT_PATH, show=False, show_infeasible=SHOW_INFEASIBLE)
        if PLOT_PATH:
            print(f"Wrote {PLOT_PATH}")
        plot_best_so_far(final, feasible,
                         title="Monte Carlo sampling of SoS1: best feasible final cumulative reward found so far",
                         path=BEST_PLOT_PATH, show=False)
        if BEST_PLOT_PATH:
            print(f"Wrote {BEST_PLOT_PATH}")
        if SHOW_PLOT:
            import matplotlib.pyplot as plt
            plt.show()
    else:
        print("No feasible points yet: nothing to plot (relax MIN_REWARD_BOUND or draw more samples)")