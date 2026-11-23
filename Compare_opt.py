"""
Runs the three SoS1 scripts (Bayesian optimization, NSGA-II, Monte Carlo) and compares them in two figures:
  1. best-so-far histories in ONE figure: one panel per stakeholder, one line per method;
  2. Pareto points and the other evaluated points of each method: one row per method (pairwise panels
     of the three stakeholders, same axis ranges in every row) plus a last row with the Pareto points of
     all methods together, showing which of them stay non-dominated when all methods are pooled.

    best so far (method m, stakeholder s, evaluation n)
        = highest final cumulative reward of s among the FEASIBLE designs of m evaluated up to n
          (infeasible designs do not count; the value stays flat until a feasible design beats it)

Each script runs as a separate process, exactly as if started on its own, with its own settings
(budget, seed, parallel processes, ...): change those settings in the scripts themselves.
Afterwards the results CSV of each script is read and the comparison is drawn.
Plot windows of the individual scripts are suppressed while they run; their PNG/PDF files are still saved.

Needs, in the same folder: sos1_bayes_opt.py, sos1_nsga2.py, sos1_monte_carlo.py,
                          sos1_evaluate.py and SoS_Markov_game_with_transitions.py
For a fair comparison give all three methods the same number of evaluations:
    BO: N_INIT + MAX_ITER     NSGA-II: POP_SIZE x N_GEN     Monte Carlo: N_SAMPLES
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

# =====================================================================================
# 1. Settings
# =====================================================================================
STAKEHOLDERS = ["Ministry of Finance", "Space agency", "Private company"]

# The three methods: script to run, the results CSV it writes (its RESULTS_CSV setting) and plot style.
# Colours: categorical slots 1-3 (checked for colour-blind separation); the marker shape repeats the
# distinction so the lines are told apart without colour too.
METHODS = [
    dict(label="Bayesian optimization", short="BO", script="Bayesian_opt_Markov_game.py",
         results_csv="sos1_bo_results.csv", color="#2a78d6", marker="o"),
    dict(label="NSGA-II", short="NSGA-II", script="NSGAII_opt_Markov_game.py",
         results_csv="sos1_nsga2_results.csv", color="#eb6834", marker="s"),
    dict(label="Monte Carlo", short="MC", script="MonteCarlo_Markov_game.py",
         results_csv="sos1_mc_results.csv", color="#1baf7a", marker="^"),
]

RUN = "all"          # "all"    : run every script, then plot
                     # "missing": run only the scripts whose results CSV does not exist yet, then plot
                     # "none"   : do not run anything, only plot from the existing results CSVs
SAVE_LOGS = True     # also write each script's console output to sos1_compare_<method>.log

PLOT_PATH = None    # combined best-so-far plot (also saved as PDF); None = don't save
PARETO_PLOT_PATH = None   # Pareto comparison plot (also saved as PDF); None = don't save
SHOW_INFEASIBLE = True                         # also draw infeasible points (faint) in the Pareto plot
COMPARE_CSV = None   # best-so-far histories of all methods; None = don't write
SHOW_PLOT = True                               # open the plot windows at the end

PLOT_STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df",
              "feasible": "#8a8984", "infeasible": "#c9c8c3"}

HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================================
# 2. Running a script
# =====================================================================================
def run_script(script, log_path=None):
    """Runs one script in its own Python process (same interpreter, this folder) and streams its output
    to the console (and to log_path). Returns True if it finished without error."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")   # no plot windows; files still saved
    print(f"\n{'=' * 90}\nRunning {script}\n{'=' * 90}", flush=True)
    t0 = time.time()
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen([sys.executable, "-u", script], cwd=HERE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            if "non-interactive" in line and "FigureCanvasAgg" in line:      # warning from plt.show() under Agg
                continue
            print(line, end="", flush=True)
            if log:
                log.write(line)
        proc.wait()
    finally:
        if log:
            log.close()
    ok = proc.returncode == 0
    print(f"--> {script} {'finished' if ok else f'FAILED (exit code {proc.returncode})'} "
          f"after {(time.time() - t0) / 60:.1f} min", flush=True)
    return ok


# =====================================================================================
# 3. Reading the results and computing the best-so-far history
# =====================================================================================
def best_so_far(final, feasible):
    """(n, k) running maximum of the final rewards over the feasible evaluations (NaN before the first one)."""
    values = np.where(np.asarray(feasible)[:, None], final, -np.inf)
    best = np.maximum.accumulate(values, axis=0)
    best[np.isneginf(best)] = np.nan
    return best


def read_results(path):
    """Results CSV of one method -> dict with final (n, 3), feasible (n,), best (n, 3), in evaluation order."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no evaluated designs")
    missing = [c for c in [f"final | {s}" for s in STAKEHOLDERS] + ["feasible"] if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    order = np.argsort([int(r["point"]) for r in rows]) if "point" in rows[0] else np.arange(len(rows))
    rows = [rows[i] for i in order]
    final = np.array([[float(r[f"final | {s}"]) for s in STAKEHOLDERS] for r in rows])
    feasible = np.array([r["feasible"].strip().lower() == "true" for r in rows])
    return {"final": final, "feasible": feasible, "best": best_so_far(final, feasible)}


def pareto_mask(F):
    """Non-dominated rows of F (all objectives minimized)."""
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if keep[i]:
            dominated = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
            keep[dominated] = False
    return keep


def add_pareto_info(results):
    """Adds to each method's data: 'pareto' (indices of its own Pareto-optimal feasible designs) and
    'overall' (bool per Pareto index: still non-dominated when the feasible designs of ALL methods are pooled).
    Returns the pooled feasible final rewards (used for the hypervolume normalization)."""
    pooled, owner = [], []
    for j, (_, d) in enumerate(results):
        feas = np.where(d["feasible"])[0]
        d["pareto"] = feas[pareto_mask(-d["final"][feas])] if len(feas) else np.array([], dtype=int)
        pooled.append(d["final"][feas])
        owner += [(j, i) for i in feas]
    pooled = np.vstack(pooled) if owner else np.empty((0, len(STAKEHOLDERS)))
    on_front = set()
    if len(pooled):
        on_front = {owner[t] for t in np.where(pareto_mask(-pooled))[0]}
    for j, (_, d) in enumerate(results):
        d["overall"] = np.array([(j, i) in on_front for i in d["pareto"]], dtype=bool)
    return pooled


def hypervolume(final, pooled):
    """Hypervolume of one method's feasible final rewards (larger = better front: closer and wider).
    Rewards are scaled to [0, 1] using the best and worst feasible values of ALL methods pooled, and the
    reference point is 1.1 x worst, so the values of the methods are comparable to each other.
    Returns None if pymoo is not installed or there are no feasible designs."""
    try:
        from pymoo.indicators.hv import HV
    except ImportError:
        return None
    if len(final) == 0 or len(pooled) == 0:
        return 0.0 if len(pooled) else None
    best, worst = pooled.max(axis=0), pooled.min(axis=0)
    scale = np.where(best - worst > 1e-12, best - worst, 1.0)
    F = (best - final) / scale                       # 0 = best pooled value, 1 = worst; minimized
    return float(HV(ref_point=np.full(final.shape[1], 1.1))(F))


# =====================================================================================
# 4. Combined best-so-far plot
# =====================================================================================
def _spread(values, min_gap):
    """Positions for end labels: same order as values, at least min_gap apart, as close as possible."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    shift = (pos - np.array(values, dtype=float)[order]).mean()           # re-centre the group
    pos -= shift
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def plot_comparison(results, path=None, show=True):
    """results: list of (method dict, data dict). One panel per stakeholder, one step line per method."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n_max = max(len(d["final"]) for _, d in results)
    fig, axes = plt.subplots(1, len(STAKEHOLDERS), figsize=(15, 4.9))
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

        ends = []
        for m, d in results:
            best = d["best"][:, k]
            n = len(best)
            x = np.arange(1, n + 1)
            ax.plot(np.r_[x, n + 0.5], np.r_[best, best[-1]], drawstyle="steps-post", color=m["color"],
                    linewidth=2, zorder=3)
            b = np.where(np.isnan(best), -np.inf, best)
            new = np.r_[b[0] > -np.inf, b[1:] > b[:-1]]                    # evaluations that set a new best
            ax.scatter(x[new], best[new], s=34, marker=m["marker"], color=m["color"],
                       edgecolors=PLOT_STYLE["surface"], linewidths=1.2, zorder=4)
            if not np.isnan(best[-1]):
                ends.append((m, float(best[-1])))

        ax.set_xlim(0.5, n_max + 0.5)
        ax.margins(y=0.12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # direct labels at the right edge, pushed apart when the final values are close
        if ends:
            lo, hi = ax.get_ylim()
            ypos = _spread([v for _, v in ends], 0.075 * (hi - lo))
            for (m, v), y in zip(ends, ypos):
                ax.annotate(f"{m['short']} {v:.2f}", xy=(1.0, v), xycoords=("axes fraction", "data"),
                            xytext=(1.02, y), textcoords=("axes fraction", "data"), va="center",
                            fontsize=8.5, color=PLOT_STYLE["text"], annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=m["color"], linewidth=1, shrinkA=0, shrinkB=2))
        ax.set_title(sid, loc="left", fontsize=11, color=PLOT_STYLE["text"])
        ax.set_xlabel("Function evaluation", fontsize=10, color=PLOT_STYLE["text2"])
    axes[0].set_ylabel("Best feasible final reward so far", fontsize=10, color=PLOT_STYLE["text2"])

    handles = [plt.Line2D([], [], color=m["color"], linewidth=2, marker=m["marker"], markersize=6,
                          markeredgecolor=PLOT_STYLE["surface"]) for m, _ in results]
    labels = [f"{m['label']} ({len(d['final'])} evaluations, {int(d['feasible'].sum())} feasible)"
              for m, d in results]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=PLOT_STYLE["text"])
    fig.suptitle("SoS1: best feasible final cumulative reward found so far, by method",
                 x=0.01, ha="left", fontsize=13, color=PLOT_STYLE["text"])
    fig.text(0.01, 0.015, "Each stakeholder's maximum is taken separately; infeasible evaluations do not count; "
                          "markers show evaluations that set a new best; a line starts at its first feasible "
                          "evaluation.", fontsize=8.5, color=PLOT_STYLE["text2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.88), w_pad=5)
    if path:
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(path.rsplit(".", 1)[0] + ".pdf", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# =====================================================================================
# 5. Pareto comparison plot
# =====================================================================================
def _style_axes(ax):
    ax.set_facecolor(PLOT_STYLE["surface"])
    ax.grid(True, color=PLOT_STYLE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PLOT_STYLE["grid"])
    ax.tick_params(colors=PLOT_STYLE["text2"], labelsize=8.5)


def plot_pareto_comparison(results, path=None, show=True, show_infeasible=True):
    """Rows 1..M: one method each -> its Pareto points (method colour and marker), its other feasible
    points (grey) and, optionally, its infeasible points (faint x). Last row: the Pareto points of all
    methods together; filled = still non-dominated when all methods are pooled, hollow = dominated by a
    design another method found. Columns: the three pairs of stakeholders, same axis ranges in every row."""
    import matplotlib.pyplot as plt
    pairs = [(0, 1), (0, 2), (1, 2)]
    n_rows = len(results) + 1

    # common axis range per stakeholder, from every point that is drawn
    shown = [d["final"][d["feasible"] | (show_infeasible & ~d["feasible"])] for _, d in results]
    allpts = np.vstack([p for p in shown if len(p)])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    pad = np.where(hi - lo > 1e-12, 0.07 * (hi - lo), 0.5)
    lims = [(lo[k] - pad[k], hi[k] + pad[k]) for k in range(len(STAKEHOLDERS))]

    fig, axes = plt.subplots(n_rows, len(pairs), figsize=(15, 3.7 * n_rows + 1.3), squeeze=False)
    fig.patch.set_facecolor(PLOT_STYLE["surface"])
    for r in range(n_rows):
        for c, (i, j) in enumerate(pairs):
            ax = axes[r, c]
            _style_axes(ax)
            if r < len(results):                                          # one method
                m, d = results[r]
                feas, par = d["feasible"], np.zeros(len(d["final"]), dtype=bool)
                par[d["pareto"]] = True
                if show_infeasible and (~feas).any():
                    ax.scatter(d["final"][~feas, i], d["final"][~feas, j], s=26, marker="x", linewidths=1,
                               color=PLOT_STYLE["infeasible"], zorder=2)
                other = feas & ~par
                ax.scatter(d["final"][other, i], d["final"][other, j], s=28, color=PLOT_STYLE["feasible"],
                           alpha=0.7, edgecolors=PLOT_STYLE["surface"], linewidths=0.8, zorder=3)
                ax.scatter(d["final"][par, i], d["final"][par, j], s=62, marker=m["marker"], color=m["color"],
                           edgecolors=PLOT_STYLE["surface"], linewidths=1.3, zorder=4)
                if c == 0:
                    ax.set_title(f"{m['label']}: {len(d['pareto'])} Pareto-optimal of {int(feas.sum())} feasible "
                                 f"({len(feas)} evaluated)", loc="left", fontsize=10.5, color=PLOT_STYLE["text"])
            else:                                                         # all methods together
                for m, d in results:
                    P = d["final"][d["pareto"]]
                    ov = d["overall"]
                    ax.scatter(P[ov, i], P[ov, j], s=62, marker=m["marker"], color=m["color"],
                               edgecolors=PLOT_STYLE["surface"], linewidths=1.3, zorder=4)
                    ax.scatter(P[~ov, i], P[~ov, j], s=52, marker=m["marker"], facecolors=PLOT_STYLE["surface"],
                               edgecolors=m["color"], linewidths=1.5, zorder=3)
                if c == 0:
                    ax.set_title("All methods: Pareto points (filled = still non-dominated when all methods "
                                 "are pooled)", loc="left", fontsize=10.5, color=PLOT_STYLE["text"])
            ax.set_xlim(*lims[i])
            ax.set_ylim(*lims[j])
            ax.set_ylabel(f"{STAKEHOLDERS[j]}", fontsize=9, color=PLOT_STYLE["text2"])
            if r == n_rows - 1:
                ax.set_xlabel(f"{STAKEHOLDERS[i]}", fontsize=9, color=PLOT_STYLE["text2"])

    handles, labels = [], []
    for m, _ in results:
        handles.append(plt.Line2D([], [], linestyle="", marker=m["marker"], markersize=8, color=m["color"],
                                  markeredgecolor=PLOT_STYLE["surface"]))
        labels.append(f"{m['label']}: Pareto-optimal")
    handles.append(plt.Line2D([], [], linestyle="", marker="o", markersize=6, color=PLOT_STYLE["feasible"],
                              alpha=0.7))
    labels.append("Feasible, not Pareto-optimal")
    if show_infeasible:
        handles.append(plt.Line2D([], [], linestyle="", marker="x", markersize=6, markeredgewidth=1,
                                  color=PLOT_STYLE["infeasible"]))
        labels.append("Infeasible")
    handles.append(plt.Line2D([], [], linestyle="", marker="o", markersize=7, markerfacecolor=PLOT_STYLE["surface"],
                              markeredgecolor=PLOT_STYLE["text2"], markeredgewidth=1.5))
    labels.append("Dominated by another method (last row)")
    top = 1 - 0.95 / fig.get_figheight()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, top), ncol=len(labels),
               frameon=False, fontsize=8.5, labelcolor=PLOT_STYLE["text"], handletextpad=0.3, columnspacing=1.2)
    fig.suptitle("SoS1: final cumulative reward of each stakeholder, by method (higher is better)",
                 x=0.01, y=1 - 0.3 / fig.get_figheight(), ha="left", fontsize=13, color=PLOT_STYLE["text"])
    fig.text(0.01, 0.4 / fig.get_figheight(),
             "Pareto-optimal over all three stakeholders together (within each method), so in a two-stakeholder "
             "panel a Pareto point can appear dominated. Axis ranges are the same in every row.",
             fontsize=8.5, color=PLOT_STYLE["text2"])
    fig.tight_layout(rect=(0, 0.8 / fig.get_figheight(), 1, 1 - 1.35 / fig.get_figheight()), w_pad=2.5, h_pad=1.5)
    if path:
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(path.rsplit(".", 1)[0] + ".pdf", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# =====================================================================================
# 6. Main
# =====================================================================================
if __name__ == "__main__":
    if RUN not in ("all", "missing", "none"):
        raise ValueError("RUN must be 'all', 'missing' or 'none'")
    os.chdir(HERE)

    # ---- run the scripts
    failed = set()
    for m in METHODS:
        exists = os.path.exists(m["results_csv"])
        if RUN == "all" or (RUN == "missing" and not exists):
            if exists:
                os.remove(m["results_csv"])        # so a failed run cannot be mistaken for a new result
            log = f"sos1_compare_{m['short'].replace('-', '').lower()}.log" if SAVE_LOGS else None
            if not run_script(m["script"], log):
                failed.add(m["label"])
        elif RUN == "missing":
            print(f"{m['label']}: using existing {m['results_csv']}")

    # ---- read the results
    results = []
    for m in METHODS:
        if not os.path.exists(m["results_csv"]):
            print(f"{m['label']}: no {m['results_csv']}"
                  f"{' (the script failed)' if m['label'] in failed else ''}; left out of the comparison")
            continue
        results.append((m, read_results(m["results_csv"])))
    if not results:
        sys.exit("No results to compare.")

    # ---- summary
    print(f"\n{'=' * 90}\nComparison of best feasible final rewards\n{'=' * 90}")
    print(f"{'method':<24}{'evals':>7}{'feasible':>10}" + "".join(f"{s:>24}" for s in STAKEHOLDERS))
    for m, d in results:
        cells = []
        for k in range(len(STAKEHOLDERS)):
            col = np.where(d["feasible"], d["final"][:, k], np.nan)
            if np.all(np.isnan(col)):
                cells.append(f"{'-':>24}")
            else:
                i = int(np.nanargmax(col))
                cells.append(f"{col[i]:>12.2f} (eval {i + 1:>4})")
        print(f"{m['label']:<24}{len(d['final']):>7}{int(d['feasible'].sum()):>10}" + "".join(cells))
    pooled = add_pareto_info(results)
    print(f"\n{'method':<24}{'Pareto-optimal':>16}{'still non-dominated':>21}{'hypervolume':>14}")
    for m, d in results:
        hv = hypervolume(d["final"][d["feasible"]], pooled)
        print(f"{m['label']:<24}{len(d['pareto']):>16}{int(d['overall'].sum()):>21}"
              f"{'-' if hv is None else f'{hv:.4f}':>14}")
    print("  still non-dominated: Pareto points of the method that no feasible design of any method dominates")
    print("  hypervolume: of the method's feasible points, rewards scaled with the best/worst feasible values "
          "of all methods\n  (reference point 1.1 x worst); larger = closer to the best values and wider spread")
    n_evals = {len(d["final"]) for _, d in results}
    if len(n_evals) > 1:
        print("\nNote: the methods used different numbers of evaluations; for a fair comparison give them the "
              "same budget (BO: N_INIT + MAX_ITER, NSGA-II: POP_SIZE x N_GEN, Monte Carlo: N_SAMPLES).")

    # ---- combined CSV: one row per method and evaluation
    if COMPARE_CSV:
        with open(COMPARE_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "evaluation", "feasible"] + [f"final | {s}" for s in STAKEHOLDERS]
                       + [f"best so far | {s}" for s in STAKEHOLDERS])
            for m, d in results:
                for i in range(len(d["final"])):
                    w.writerow([m["label"], i + 1, bool(d["feasible"][i])] + list(d["final"][i])
                               + ["" if np.isnan(v) else v for v in d["best"][i]])
        print(f"\nWrote {COMPARE_CSV}")

    # ---- combined plots
    if any(d["feasible"].any() for _, d in results):
        plot_comparison(results, path=PLOT_PATH, show=False)
        if PLOT_PATH:
            print(f"Wrote {PLOT_PATH}")
        plot_pareto_comparison(results, path=PARETO_PLOT_PATH, show=False, show_infeasible=SHOW_INFEASIBLE)
        if PARETO_PLOT_PATH:
            print(f"Wrote {PARETO_PLOT_PATH}")
        if SHOW_PLOT:
            import matplotlib.pyplot as plt
            plt.show()
    else:
        print("No feasible designs in any method: nothing to plot")