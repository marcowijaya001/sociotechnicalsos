"""
Multi-objective constrained optimization of SoS1 with NSGA-II (pymoo).

    design variables : selected own-reward values and transfer-rule amounts: continuous (Real), integer
                       (Integer) or one of a list of levels (Choice)
    objectives       : final cumulative reward of each stakeholder        -> maximize (all three)
    constraints      : minimum cumulative reward of each stakeholder      -> must stay >= a lower bound

Every design point is evaluated with evaluate_sos1 (Nash Q-value iteration + best-policy sequence).

Needs, in the same folder: sos1_evaluate.py and SoS_Markov_game_with_transitions.py
Needs pymoo:  pip install -U pymoo
"""
import csv
import json
import multiprocessing
import time

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
from pymoo.optimize import minimize
from pymoo.parallelization.starmap import StarmapParallelization

from function_for_optimization import evaluate_sos1

# =====================================================================================
# 1. Settings
# =====================================================================================
STAKEHOLDERS = ["Ministry of Finance", "Space agency", "Private company"]

# Design variables: (kind, where, variable)
#   kind "rule": where = (rule name,)                       for "flow" rules (amount moved)
#                where = (rule name, stakeholder)           for "transfers" rules
#   kind "own" : where = (stakeholder, state, action)       own reward of that action in that state
#   variable   : Real(bounds=(lower, upper))                continuous
#                Integer(bounds=(lower, upper))             whole numbers lower, ..., upper
#                Choice(options=[v1, v2, ...])              one of a list of numeric levels
#                                                           (NSGA-II treats the levels as unordered)
DESIGN_VARIABLES = [
    ("rule", ("funding #1",),                                   Integer(bounds=(3, 8))),
    ("rule", ("system acquisition",),                           Real(bounds=(1.0, 5.0))),
    ("rule", ("technology acquisition",),                       Real(bounds=(0.5, 3.0))),
    ("rule", ("service transaction system 1", "Space agency"),  Choice(options=[0.0, 0.25, 0.5, 0.75, 1.0])),
    ("own",  ("Ministry of Finance", "Data Collection", "use"), Real(bounds=(1.0, 4.0))),
    ("own",  ("Private company", "High Battery, Operation Satellite #2", "operate 2"), Real(bounds=(-1.0, 0.0))),
]

# Constraints: minimum cumulative reward over the time steps must stay >= these bounds
MIN_REWARD_BOUND = {"Ministry of Finance": -12.0, "Space agency": -2.0, "Private company": -2.0}

# Evaluation settings (passed to evaluate_sos1)
EVAL_SETTINGS = dict(beta=0.99, n_iter=10, n_steps=8)

# NSGA-II settings: POP_SIZE x N_GEN evaluations in total (about 30 s each with n_iter=50)
POP_SIZE = 2
N_GEN = 4
SEED = 1
N_PROCESSES = 1       # > 1: evaluate a generation in parallel worker processes (e.g. number of CPU cores)

RESULTS_CSV = "sos1_nsga2_results.csv"      # all evaluated points; None = don't write
SEQUENCES_CSV = "sos1_nsga2_sequences.csv"  # step-by-step best-policy sequences; None = don't write
SEQUENCES_FOR = "pareto"                    # "pareto": only Pareto-optimal designs, "all": every evaluated design
PLOT_PATH = "sos1_nsga2_pareto.png"         # Pareto plot (also saved as PDF); None = don't save
BEST_PLOT_PATH = "sos1_nsga2_best_so_far.png"  # best feasible final reward so far vs evaluations (also PDF); None = don't save
SHOW_PLOT = True                            # open the plot window
SHOW_INFEASIBLE = True                      # also draw infeasible points (faint), for context


# =====================================================================================
# 2. From design values to the inputs of evaluate_sos1
# =====================================================================================
NAMES = [" / ".join(where) for _, where, _ in DESIGN_VARIABLES]
KEYS = [f"x{i}" for i in range(len(DESIGN_VARIABLES))]          # variable names used by pymoo
ALL_REAL = all(isinstance(var, Real) for _, _, var in DESIGN_VARIABLES)


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


def design_values(X):
    """pymoo individual -> list of design values (X is an array if all variables are Real, else a dict)."""
    if isinstance(X, dict):
        return [float(X[k]) for k in KEYS]
    return [float(v) for v in X]


# =====================================================================================
# 3. The optimization problem
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


class SoS1Problem(ElementwiseProblem):
    """F: negative final rewards (pymoo minimizes); G: MIN_REWARD_BOUND - minimum reward (feasible if <= 0)."""

    def __init__(self, **kwargs):
        n = len(STAKEHOLDERS)
        if ALL_REAL:      # continuous problem: plain bounds -> standard NSGA-II operators
            xl = np.array([var.bounds[0] for _, _, var in DESIGN_VARIABLES], dtype=float)
            xu = np.array([var.bounds[1] for _, _, var in DESIGN_VARIABLES], dtype=float)
            super().__init__(n_var=len(DESIGN_VARIABLES), n_obj=n, n_ieq_constr=n, xl=xl, xu=xu, **kwargs)
        else:             # mixed problem: named variables
            super().__init__(vars={k: var for k, (_, _, var) in zip(KEYS, DESIGN_VARIABLES)},
                             n_obj=n, n_ieq_constr=n, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t0 = time.time()
        final, minimum, sequence = evaluate_design(design_values(X))
        out["F"] = [-final[s] for s in STAKEHOLDERS]
        out["G"] = [MIN_REWARD_BOUND[s] - minimum[s] for s in STAKEHOLDERS]
        out["SEQUENCE"] = json.dumps(sequence)          # kept with every evaluated individual
        out["EVAL_TIME"] = time.time() - t0


class CollectEvaluations(Callback):
    """Keeps every evaluated design (pymoo's result only holds the final population)."""

    def __init__(self):
        super().__init__()
        self.X, self.F, self.G, self.SEQ, self.GEN = [], [], [], [], []

    def notify(self, algorithm):
        pop = algorithm.pop if algorithm.n_gen == 1 or algorithm.off is None else algorithm.off
        self.X += [design_values(x) for x in pop.get("X")]
        self.F += list(pop.get("F"))
        self.G += list(pop.get("G"))
        self.SEQ += [json.loads(s) for s in pop.get("SEQUENCE")]
        self.GEN += [algorithm.n_gen] * len(pop)          # generation of each evaluation
        print(f"  generation {algorithm.n_gen:>3}: {len(self.X)} designs evaluated so far")


# =====================================================================================
# 4. Pareto front of the feasible evaluated points
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
# 5. Plot: Pareto points and all feasible points, pairwise for the three objectives
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
    fig.suptitle("NSGA-II optimization of SoS1: final cumulative reward of each stakeholder (higher is better)",
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
# 6. Optimization
# =====================================================================================
if __name__ == "__main__":
    pool = multiprocessing.Pool(N_PROCESSES) if N_PROCESSES > 1 else None
    runner_kwargs = {"elementwise_runner": StarmapParallelization(pool.starmap)} if pool else {}
    problem = SoS1Problem(**runner_kwargs)

    if ALL_REAL:
        algorithm = NSGA2(pop_size=POP_SIZE)                                      # SBX crossover, polynomial mutation
    else:
        algorithm = MixedVariableGA(pop_size=POP_SIZE, survival=RankAndCrowdingSurvival())   # NSGA-II survival

    collect = CollectEvaluations()
    print(f"NSGA-II: {len(DESIGN_VARIABLES)} design variables ({'continuous' if ALL_REAL else 'mixed'}), "
          f"{len(STAKEHOLDERS)} objectives, {len(STAKEHOLDERS)} constraints, "
          f"population {POP_SIZE} x {N_GEN} generations, {N_PROCESSES} process(es)")
    t0 = time.time()
    res = minimize(problem, algorithm, ("n_gen", N_GEN), seed=SEED, callback=collect, verbose=True)
    if pool:
        pool.close()
        pool.join()
    print(f"\nDone: {len(collect.X)} SoS1 evaluations in {(time.time() - t0) / 60:.1f} min")

    # ---- results over ALL evaluated designs
    V = np.array(collect.X)                        # (n, n_var) design values
    final = -np.array(collect.F)                   # (n, 3) final cumulative rewards (sign undone)
    G = np.array(collect.G)
    minimum = np.array([[MIN_REWARD_BOUND[s] for s in STAKEHOLDERS]]) - G     # (n, 3) minimum rewards
    feasible = np.all(G <= 1e-9, axis=1)
    sequences = collect.SEQ
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
        gen = collect.GEN
        boundaries = [i + 0.5 for i in range(1, len(gen)) if gen[i] != gen[i - 1]]
        plot_best_so_far(final, feasible, title="NSGA-II optimization of SoS1: best feasible final cumulative reward found so far",
                         path=BEST_PLOT_PATH, show=False, boundaries=boundaries,
                         boundary_note="Dashed lines: boundaries between generations.")
        if BEST_PLOT_PATH:
            print(f"Wrote {BEST_PLOT_PATH}")
        if SHOW_PLOT:
            import matplotlib.pyplot as plt
            plt.show()
    else:
        print("No feasible points yet: nothing to plot (relax MIN_REWARD_BOUND or run more generations)")