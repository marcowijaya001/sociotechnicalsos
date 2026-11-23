"""
Multi-objective constrained Bayesian optimization of SoS1 with SMT-optim (MO-SEGO).

    design variables : selected own-reward values and transfer-rule amounts, each continuous (within bounds),
                       integer, or chosen from a list of allowed levels
    objectives       : final cumulative reward of each stakeholder        -> maximize (all three)
    constraints      : minimum cumulative reward of each stakeholder      -> must stay >= a lower bound

Every design point is evaluated with evaluate_sos1 (Nash Q-value iteration + best-policy sequence).

Needs, in the same folder: sos1_evaluate.py and SoS_Markov_game_with_transitions.py
Needs SMT-optim with multi-objective support (development version from GitHub) and its dependencies:
    pip install git+https://github.com/SMTorg/smt-optim.git moocore pymoo
"""
import csv
import time

import numpy as np

from smt.design_space import DesignSpace, FloatVariable, IntegerVariable, OrdinalVariable, CategoricalVariable
from smt_optim.core import ObjectiveConfig, ConstraintConfig, DriverConfig, Driver, Problem
from smt_optim.surrogate_models.smt import SmtGPX, SmtAutoModel
from smt_optim.acquisition_strategies.mosego import MOSEGO
from smt_optim.acquisition_functions import init_mpi

from function_for_optimization import evaluate_sos1

# =====================================================================================
# 1. Settings
# =====================================================================================
STAKEHOLDERS = ["Ministry of Finance", "Space agency", "Private company"]

# Design variables: (kind, where, variable)
#   kind "rule": where = (rule name,)                       for "flow" rules (amount moved)
#                where = (rule name, stakeholder)           for "transfers" rules
#   kind "own" : where = (stakeholder, state, action)       own reward of that action in that state
#   variable   : FloatVariable(lower, upper)                continuous
#                IntegerVariable(lower, upper)              whole numbers lower, lower+1, ..., upper
#                OrdinalVariable([v1, v2, ...])             one of a list of ordered numeric levels
#                CategoricalVariable([v1, v2, ...])         one of a list of numeric levels without an order
DESIGN_VARIABLES = [
    ("rule", ("funding #1",),                                   IntegerVariable(3, 8)),
    ("rule", ("system acquisition",),                           FloatVariable(1.0, 5.0)),
    ("rule", ("technology acquisition",),                       FloatVariable(0.5, 3.0)),
    ("rule", ("service transaction system 1", "Space agency"),  OrdinalVariable([0.0, 0.25, 0.5, 0.75, 1.0])),
    ("own",  ("Ministry of Finance", "Data Collection", "use"), FloatVariable(1.0, 4.0)),
    ("own",  ("Private company", "High Battery, Operation Satellite #2", "operate 2"), FloatVariable(-1.0, 0.0)),
]

# Constraints: minimum cumulative reward over the time steps must stay >= these bounds
MIN_REWARD_BOUND = {"Ministry of Finance": -12.0, "Space agency": -2.0, "Private company": -2.0}

# Evaluation settings (passed to evaluate_sos1)
EVAL_SETTINGS = dict(beta=0.99, n_iter=10, n_steps=8)

# Bayesian optimization settings
N_INIT = 3           # initial design (Latin hypercube) points; rule of thumb ~2 x number of variables + 1
MAX_ITER = 5         # Bayesian optimization iterations after the initial design
SEED = 0
CTOL = 1e-4           # constraint tolerance to call a point feasible
RESULTS_CSV = "sos1_bo_results.csv"     # all evaluated points; None = don't write
PLOT_PATH = "sos1_bo_pareto.png"         # Pareto plot (also saved as PDF); None = don't save
BEST_PLOT_PATH = "sos1_bo_best_so_far.png"  # best feasible final reward so far vs evaluations (also PDF); None = don't save
SHOW_PLOT = True                         # open the plot window
SHOW_INFEASIBLE = True                   # also draw infeasible points (faint), for context
SEQUENCES_CSV = "sos1_bo_sequences.csv"  # step-by-step best-policy sequences; None = don't write
SEQUENCES_FOR = "pareto"                 # "pareto": only Pareto-optimal designs, "all": every evaluated design


# =====================================================================================
# 2. From a design vector x to the inputs of evaluate_sos1
# =====================================================================================
def decode(x):
    """Optimizer vector -> actual design values. SMT-optim passes integer variables as numbers and
    ordinal/categorical variables as the INDEX of the chosen level; this returns the level itself."""
    values = []
    for v, (_, _, var) in zip(x, DESIGN_VARIABLES):
        if isinstance(var, IntegerVariable):
            values.append(float(int(round(float(v)))))
        elif isinstance(var, (OrdinalVariable, CategoricalVariable)):
            values.append(float(var.values[int(round(float(v)))]))
        else:
            values.append(float(v))
    return values


def to_inputs(x):
    """x (optimizer vector, same order as DESIGN_VARIABLES)
    -> own_rewards, rule_amounts dictionaries for evaluate_sos1."""
    own_rewards, rule_amounts = {}, {}
    for value, (kind, where, _) in zip(decode(x), DESIGN_VARIABLES):
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
# 3. One SoS1 evaluation per design point, shared by all objectives and constraints
# =====================================================================================
class CachedSoS1:
    """SMT-optim calls every objective and constraint function separately with the same x.
    The expensive SoS1 evaluation is done once per x and reused."""

    def __init__(self):
        self.cache = {}              # design point -> (final, minimum)
        self.sequences = {}          # design point -> best-policy sequence (games, states, actions, rewards)
        self.n_evaluations = 0

    @staticmethod
    def _key(x):
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    def __call__(self, x):
        key = self._key(x)
        if key not in self.cache:
            t0 = time.time()
            own_rewards, rule_amounts = to_inputs(x)
            final, minimum, details = evaluate_sos1(own_rewards=own_rewards, rule_amounts=rule_amounts,
                                                    return_details=True, **EVAL_SETTINGS)
            traj = details["trajectory"]
            self.cache[key] = (final, minimum)
            self.sequences[key] = {
                "games": list(traj.games),             # stage-game names s_0 ... s_T, e.g. "0_0_0"
                "states": list(traj.states),           # joint states (one state name per stakeholder)
                "actions": list(traj.actions),         # joint action at each step
                "rewards": traj.rewards.copy(),        # (T, 3) reward per step
                "converged": details["converged"],     # Nash-Q iteration converged within n_iter
            }
            self.n_evaluations += 1
            print(f"  evaluation {self.n_evaluations:>3} ({time.time() - t0:5.1f} s): "
                  f"final {[round(final[s], 2) for s in STAKEHOLDERS]}  "
                  f"min {[round(minimum[s], 2) for s in STAKEHOLDERS]}")
        return self.cache[key]

    def sequence(self, x):
        """Best-policy sequence of an evaluated design point (nearest stored point if not exact)."""
        key = self._key(x)
        if key not in self.sequences:
            keys = list(self.sequences)
            key = keys[int(np.argmin([np.linalg.norm(np.subtract(k, key)) for k in keys]))]
        return self.sequences[key]


sos1 = CachedSoS1()


def make_objective(sid):
    # SMT-optim's multi-objective strategy minimizes -> minimize the NEGATIVE final reward
    def objective(x):
        return -sos1(x)[0][sid]
    objective.__name__ = f"neg_final_{sid.replace(' ', '_')}"
    return objective


def make_constraint(sid):
    def constraint(x):
        return sos1(x)[1][sid]           # minimum cumulative reward, must be >= MIN_REWARD_BOUND[sid]
    constraint.__name__ = f"min_{sid.replace(' ', '_')}"
    return constraint


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
    fig.suptitle("Bayesian optimization of SoS1: final cumulative reward of each stakeholder (higher is better)",
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
    names = [" / ".join(where) for _, where, _ in DESIGN_VARIABLES]
    variables = [var for _, _, var in DESIGN_VARIABLES]

    if all(isinstance(v, FloatVariable) for v in variables):
        # all continuous: plain bounds, standard GP surrogate, Latin hypercube initial design
        design_space = np.array([[v.lower, v.upper] for v in variables], dtype=float)
        surrogate, xt_init = SmtGPX, None
    else:
        # mixed: SMT design space, surrogate with a mixed-variable kernel, and an initial design of valid
        # points (the built-in initial design does not round integer/ordinal variables)
        design_space = DesignSpace(variables, seed=SEED)
        surrogate = SmtAutoModel
        xt_init = [design_space.sample_valid_x(N_INIT, seed=SEED)[0]]

    obj_configs = [ObjectiveConfig(objective=[make_objective(sid)], surrogate=surrogate, type="minimize")
                   for sid in STAKEHOLDERS]
    cstr_configs = [ConstraintConfig(constraint=[make_constraint(sid)], lower=MIN_REWARD_BOUND[sid],
                                     surrogate=surrogate)
                    for sid in STAKEHOLDERS]
    problem = Problem(obj_configs=obj_configs, cstr_configs=cstr_configs, design_space=design_space)

    config = DriverConfig(max_iter=MAX_ITER, nt_init=N_INIT, xt_init=xt_init, ctol=CTOL, scaling=True,
                          seed=SEED, verbose=True, results_dir=None)
    driver = Driver(problem, config, MOSEGO, strategy_kwargs={"acq_init": init_mpi, "seed": SEED})

    print(f"Bayesian optimization: {len(DESIGN_VARIABLES)} design variables, {len(STAKEHOLDERS)} objectives, "
          f"{len(STAKEHOLDERS)} constraints, {N_INIT} initial points + {MAX_ITER} iterations")
    t0 = time.time()
    state = driver.optimize()
    print(f"\nDone: {sos1.n_evaluations} SoS1 evaluations in {(time.time() - t0) / 60:.1f} min")

    # ---- results
    data = state.dataset.export_as_dict()
    X = data["x"]                                  # (n, n_var) optimizer vectors (indices for ordinal/categorical)
    V = np.array([decode(x) for x in X])           # (n, n_var) actual design values
    final = -data["obj"]                           # (n, 3) final cumulative rewards (sign undone)
    minimum = data["cstr"]                         # (n, 3) minimum cumulative rewards
    feasible = data["rscv"] <= CTOL
    print(f"{feasible.sum()} of {len(X)} evaluated points satisfy all constraints")
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
            print("             " + "; ".join(f"{n} = {v:.3g}" for n, v in zip(names, V[i])))
            sq = sos1.sequence(X[i])
            print("             games  : " + " -> ".join(sq["games"]))
            print("             actions: " + " | ".join(f"t{t} ({', '.join(a)})" for t, a in enumerate(sq["actions"])))

    if RESULTS_CSV:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["point", "evaluation"] + names + [f"final | {s}" for s in STAKEHOLDERS]
                       + [f"min | {s}" for s in STAKEHOLDERS] + ["feasible", "pareto"]
                       + [f"best so far | {s}" for s in STAKEHOLDERS] + ["game sequence"])
            for i in range(len(X)):
                w.writerow([i, i + 1] + list(V[i]) + list(final[i]) + list(minimum[i])
                           + [bool(feasible[i]), bool(i in idx)]
                           + ["" if np.isnan(v) else v for v in best[i]] + [" -> ".join(sos1.sequence(X[i])["games"])])
        print(f"\nWrote {RESULTS_CSV}")

    # ---- distinct sequences of stage games among the evaluated designs
    groups = {}
    for i in range(len(X)):
        groups.setdefault(tuple(sos1.sequence(X[i])["games"]), []).append(i)
    print(f"\n{len(groups)} distinct best-policy sequences of stage games among {len(X)} evaluated designs:")
    for seq, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n_feas = int(feasible[pts].sum())
        n_par = len(set(pts) & set(idx.tolist()))
        print(f"  {len(pts):>3} designs ({n_feas} feasible, {n_par} Pareto): " + " -> ".join(seq))

    if SEQUENCES_CSV:
        chosen = idx if SEQUENCES_FOR == "pareto" else np.arange(len(X))
        with open(SEQUENCES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["point", "pareto", "step", "game"] + [f"state | {s}" for s in STAKEHOLDERS]
                       + [f"action | {s}" for s in STAKEHOLDERS] + [f"reward | {s}" for s in STAKEHOLDERS]
                       + [f"cum_reward | {s}" for s in STAKEHOLDERS] + ["next game"])
            for i in chosen:
                sq = sos1.sequence(X[i])
                cum = np.cumsum(sq["rewards"], axis=0)
                for t, a in enumerate(sq["actions"]):
                    w.writerow([i, bool(i in idx), t, sq["games"][t]] + list(sq["states"][t]) + list(a)
                               + list(sq["rewards"][t]) + list(cum[t]) + [sq["games"][t + 1]])
        print(f"Wrote {SEQUENCES_CSV} (step-by-step sequences of the "
              f"{'Pareto-optimal' if SEQUENCES_FOR == 'pareto' else 'evaluated'} designs)")

    # ---- plots: Pareto points and all feasible points; best feasible final reward so far
    if feasible.any():
        plot_pareto(final, feasible, idx, path=PLOT_PATH, show=False, show_infeasible=SHOW_INFEASIBLE)
        if PLOT_PATH:
            print(f"Wrote {PLOT_PATH}")
        boundaries = [N_INIT + 0.5] if len(X) > N_INIT else []
        plot_best_so_far(final, feasible, title="Bayesian optimization of SoS1: best feasible final cumulative reward found so far",
                         path=BEST_PLOT_PATH, show=False, boundaries=boundaries,
                         boundary_note="Dashed line: end of the initial design; Bayesian optimization iterations to the right.")
        if BEST_PLOT_PATH:
            print(f"Wrote {BEST_PLOT_PATH}")
        if SHOW_PLOT:
            import matplotlib.pyplot as plt
            plt.show()
    else:
        print("No feasible points yet: nothing to plot (relax MIN_REWARD_BOUND or run more iterations)")