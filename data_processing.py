import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
AGENTS = ["Ministry of Finance", "Space agency", "Private company"]   # agent 0, 1, 2

best_iteration = [np.array([0,-5,-10,-8.3,-6.6,-4.9,-3.2,-1.5,0.2]),
                  np.array([0,2.5,4.5,4.65,4.8,4.95,5.1,5.25,5.4]),
                  np.array([0,0.8,3.5,3,3,3,3,3,3])]
best_learning = [np.array([0,-5,-10,-8,-6.3,-4.6,-2.9,-1.2,0.5]),
                 np.array([0,4,6.5,3.5,3.65,3.8,3.95,4.1,4.25]),
                 np.array([0,-0.7,0.5,3,3,3,3,3,3])]
random_mean = [np.array([0,-3,-6.02,-8.09,-9.2,-9.81,-9.98,-10,-9.85]),
               np.array([0,-0.8,-0.16,0.56,1.09,1.44,1.59,1.64,1.63]),
               np.array([0,1.08,1.61,1.8,1.7,1.6,1.52,1.47,1.44])]
random_std = [np.array([0,2,2.89,2.66,2.36,2.48,2.81,3.1,3.27]),
              np.array([0,3.36,4.07,3.86,3.26,2.68,2.16,1.71,1.46]),
              np.array([0,1.88,1.76,1.37,1.16,1.14,1.18,1.25,1.29])]
N_RUNS = 1000                       # Monte Carlo runs behind the random-policy mean and std

# ---------------------------------------------------------------- sizes derived from the data
for name, series in [("best_iteration", best_iteration), ("best_learning", best_learning),
                     ("random_mean", random_mean), ("random_std", random_std)]:
    for k, arr in enumerate(series):                 # every series must have one value per time step
        assert len(arr) == len(x), f"{name}[{k}] has {len(arr)} values, x has {len(x)}"
STEP = float(np.min(np.diff(x))) if len(x) > 1 else 1.0    # spacing between time steps
DODGE = 0.12 * STEP                 # small x-shift of the random series so its error bars don't hide the markers
PANEL_WIDTH = min(6.5, 3.8 + 0.12 * len(x))                 # wider panels for longer horizons
MAX_TICKS = 12                      # beyond this many steps, show only a subset of tick labels

# ---------------------------------------------------------------- style
STYLE = {
    "iteration": "#2a78d6",         # best policy from Q-value iteration
    "learning":  "#eb6834",         # best policy from Q-learning
    "random":    "#52514e",         # random policy (baseline, neutral)
    "surface":   "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df",
}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


def style_axis(ax):
    ax.set_facecolor(STYLE["surface"])
    ax.grid(True, axis="y", color=STYLE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(STYLE["grid"])
    ax.tick_params(colors=STYLE["text2"])
    ax.axhline(0, color=STYLE["text2"], linewidth=0.8, zorder=1)          # zero reference


def end_labels(ax, x_end, y_ends, labels, gap_frac=0.08):
    """Labels at the right end of the lines. Lines ending at (almost) the same value share one label;
    labels are spread apart so they don't overlap."""
    lo, hi = ax.get_ylim()
    gap = gap_frac * (hi - lo)
    order = list(np.argsort(y_ends))
    groups = [[order[0]]]
    for i in order[1:]:
        if y_ends[i] - y_ends[groups[-1][-1]] < 0.5 * gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    pos = np.array([np.mean([y_ends[i] for i in g]) for g in groups])
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    if pos[-1] > hi:
        pos -= pos[-1] - hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    for g, y in zip(groups, pos):
        ax.annotate(" / ".join(labels[i] for i in sorted(g)), xy=(x_end, y), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=STYLE["text"],
                    annotation_clip=False)


# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, len(AGENTS), figsize=(PANEL_WIDTH * len(AGENTS), 4.6), sharex=True)
fig.patch.set_facecolor(STYLE["surface"])

for k, (ax, name) in enumerate(zip(axes, AGENTS)):
    style_axis(ax)
    ax.errorbar(x + DODGE, random_mean[k], yerr=random_std[k], fmt="o", linestyle=(0, (4, 3)),
                color=STYLE["random"], ecolor=STYLE["random"], elinewidth=1.2, capsize=4,
                markersize=6, linewidth=1.5, zorder=2)
    ax.plot(x, best_iteration[k], color=STYLE["iteration"], linewidth=2.2, marker="o",
            markersize=7, markeredgecolor=STYLE["surface"], markeredgewidth=1.5, zorder=3)
    ax.plot(x, best_learning[k], color=STYLE["learning"], linewidth=2.2, marker="s",
            markersize=6.5, markeredgecolor=STYLE["surface"], markeredgewidth=1.5, zorder=3)

    ax.set_title(f"Agent {k}: {name}", loc="left", color=STYLE["text"], pad=10)
    ax.set_xlabel("Time step", color=STYLE["text2"])
    if len(x) <= MAX_TICKS:
        ax.set_xticks(x)
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=MAX_TICKS, integer=True))
    ax.set_xlim(x[0] - 0.3 * STEP, x[-1] + DODGE + 0.3 * STEP)
    if k == 0:
        ax.set_ylabel("Cumulative reward", color=STYLE["text2"])
    end_labels(ax, x[-1] + DODGE, [best_iteration[k][-1], best_learning[k][-1], random_mean[k][-1]],
               ["Iteration", "Learning", "Random"])

handles = [
    plt.Line2D([], [], color=STYLE["iteration"], linewidth=2.2, marker="o", markersize=7,
               markeredgecolor=STYLE["surface"], markeredgewidth=1.5),
    plt.Line2D([], [], color=STYLE["learning"], linewidth=2.2, marker="s", markersize=6.5,
               markeredgecolor=STYLE["surface"], markeredgewidth=1.5),
    plt.Line2D([], [], color=STYLE["random"], linewidth=1.5, linestyle=(0, (4, 3)), marker="o", markersize=6),
]
labels = ["Best policy, Q-value iteration",
          "Best policy, Q-value learning",
          f"Random policy: mean ± 1 std ({N_RUNS} Monte Carlo runs)"]
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.915), ncol=3,
           frameon=False, labelcolor=STYLE["text"])
fig.suptitle("Cumulative reward over time per stakeholder", x=0.01, ha="left",
             fontsize=14, color=STYLE["text"])
fig.tight_layout(rect=(0, 0, 0.96, 0.88), w_pad=3)

fig.savefig("cumulative_rewards.png", dpi=300, facecolor=fig.get_facecolor())
fig.savefig("cumulative_rewards.pdf", facecolor=fig.get_facecolor())        # vector version for the thesis
plt.show()