"""
Payoff + transition framework for Markov games, with an automated Nash Q-learning loop.

    Stakeholder           states (optionally grouped under life-cycle phases), actions,
                          actions available per state, own (self) rewards, state transitions
    ResourceTransferRule  conditional reward transfers between stakeholders,
                          triggered by their states and actions
    VisitLimit            how often a stakeholder may visit a state (entries or time steps)
    ActionLimit           how often a stakeholder may take an action (optionally only in given states)
    PrecedenceConstraint  an action / a move of a stakeholder is only valid after other
                          stakeholders have been in given states
    MarkovGame            holds stakeholders + rules + constraints, enumerates all stage games, builds
                          their reward arrays R[a_1, ..., a_n, i] and the joint transition
                          map (joint state, joint action) -> next stage game(s), and runs
                          Nash Q-learning over all stage games

Constraints that depend on the history (VisitLimit, ActionLimit, PrecedenceConstraint) are handled by state
augmentation: each stakeholder's state is extended with a small memory (visit counters, action
counters, "has been in" flags) so the game stays Markov. A stage game is then a joint (state + memory) of all stakeholders,
labelled e.g. "Funding #2 [satellite operating=no]"; stakeholders whose memory does not matter in a
state keep the plain state name. The restricted actions are removed from the stage games ("mask") or
their restricted move fails ("stay"), so the Nash solver and the Q-value iteration are unchanged:
they simply work on the (augmented) stage games.

Lookup keys used in user inputs (available actions, own rewards, transitions, rule conditions):
    a state name   e.g. "manufacture system 1"   (most specific)
    a phase name   e.g. "production"             (applies to every state in that phase)
    "*"            any state                     (least specific)

Naming convention for stage games (as in hand-written Nash-Q scripts):
    Q_game_i_j / r_game_i_j  ->  i = state index of stakeholder 0, j = state index of stakeholder 1, ...
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from itertools import combinations, product
from math import prod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import pygambit as gbt
import numpy as np

Number = Union[int, float]
JointState = Tuple[str, ...]
WILDCARD = "*"

def NashEquilibriumMixedStrategy(Q_i):
    # Converting format of array
    Q = np.asarray(Q_i)
    g = gbt.Game.from_arrays(*[Q[..., i] for i in range(Q.shape[-1])])
    # Solver for mixed strategy for finding Nash equilibriium points
    result = gbt.nash.logit_solve(g)
    for eq in result.equilibria:
        probabilities = [[float(eq[s]) for s in p.strategies] for p in g.players]  # probabilities
        Nash_values = [float(eq.payoff(p)) for p in g.players]  # expected values
        max_regret = float(eq.max_regret())

    return Nash_values


def NashEquilibriumMixedStrategyProbs(Q_i):
    """Same solver as NashEquilibriumMixedStrategy (pygambit logit_solve), but returns the
    probabilities it computes as well: (Nash_values, probabilities). The probabilities are
    needed to follow the best policy."""
    Q = np.asarray(Q_i)
    g = gbt.Game.from_arrays(*[Q[..., i] for i in range(Q.shape[-1])])
    result = gbt.nash.logit_solve(g)
    for eq in result.equilibria:
        probabilities = [[float(eq[s]) for s in p.strategies] for p in g.players]  # probabilities
        Nash_values = [float(eq.payoff(p)) for p in g.players]  # expected values

    return Nash_values, probabilities

# =========================================================================== states
@dataclass(frozen=True)
class State:
    name: str
    phase: Optional[str] = None

    def matches(self, key: str) -> bool:
        return key == WILDCARD or key == self.name or (self.phase is not None and key == self.phase)

    def __str__(self) -> str:
        return self.name if self.phase in (None, self.name) else f"{self.phase}/{self.name}"


def _parse_states(spec) -> List[State]:
    """Accepted forms:
        ["low FoM_tech", "medium FoM_tech"]                 flat state names
        ["production", "operation"]                         phases used directly as states
        {"production": ["manufacture system 1", ...],       phase -> states in that phase
         "operation":  []}                                  (empty list: the phase itself is a state)
        [State(...), ...]
    """
    if isinstance(spec, dict):
        states = []
        for phase, subs in spec.items():
            states += [State(n, phase) for n in subs] if subs else [State(phase, phase)]
        return states
    return [s if isinstance(s, State) else State(str(s)) for s in spec]


def _as_list(x) -> list:
    return list(x) if isinstance(x, (list, tuple, set)) else [x]


# =========================================================================== stakeholder
@dataclass
class Stakeholder:
    id: str
    states: Any                                            # see _parse_states
    actions: List[str]                                     # verbs, e.g. "fund", "manufacture"
    available_actions: Optional[Dict[str, List[str]]] = None
    #   {state | phase | "*": [actions]}; states with no entry get all actions
    own_rewards: Union[None, Dict[str, Dict[str, Number]], Callable[[State, str], Number]] = None
    #   {state | phase | "*": {action: reward}}  or  f(state, action) -> reward
    default_reward: Number = 0.0
    transitions: Optional[List[Dict[str, Any]]] = None
    #   [{"initial_state": state | phase | "*", "action": a, "final_state": state,
    #     "probability": p}]            "probability" is optional (default 1 = deterministic)
    #   (state, action) pairs without an entry stay in the same state, unless strict_transitions
    strict_transitions: bool = False

    def __post_init__(self):
        self.states = _parse_states(self.states)
        self.actions = list(self.actions)
        self.available_actions = dict(self.available_actions or {})
        self.own_rewards = self.own_rewards if self.own_rewards is not None else {}
        self.transitions = list(self.transitions or [])
        self._validate()
        self._next = self._build_transition_table()

    # ---- queries
    @property
    def phases(self) -> List[str]:
        return list(dict.fromkeys(s.phase for s in self.states if s.phase))

    def state(self, name: str) -> State:
        for s in self.states:
            if s.name == name:
                return s
        raise KeyError(f"[{self.id}] unknown state '{name}'")

    def state_index(self, name: str) -> int:
        return [s.name for s in self.states].index(name)

    def is_valid_key(self, key: str) -> bool:
        return any(s.matches(key) for s in self.states)

    def actions_in(self, state: State) -> List[str]:
        for key in (state.name, state.phase, WILDCARD):          # most specific first
            if key is not None and key in self.available_actions:
                return list(self.available_actions[key])
        return list(self.actions)

    def own_reward(self, state: State, action: str) -> float:
        if callable(self.own_rewards):
            return float(self.own_rewards(state, action))
        for key in (state.name, state.phase, WILDCARD):          # most specific first
            if key is not None and action in self.own_rewards.get(key, {}):
                return float(self.own_rewards[key][action])
        return float(self.default_reward)

    def next_state_distribution(self, state: State, action: str) -> Dict[str, float]:
        """{final_state_name: probability} for this stakeholder's own transition."""
        return self._next[(state.name, action)]

    # ---- transitions
    def _build_transition_table(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        table = {}
        for s in self.states:
            for a in self.actions_in(s):
                dist = None
                for key in (s.name, s.phase, WILDCARD):          # most specific first
                    if key is None:
                        continue
                    entries = [t for t in self.transitions
                               if t["initial_state"] == key and t["action"] == a]
                    if entries:
                        dist = {}
                        for t in entries:
                            dist[t["final_state"]] = dist.get(t["final_state"], 0.0) + float(t.get("probability", 1.0))
                        total = sum(dist.values())
                        if abs(total - 1.0) > 1e-9:
                            raise ValueError(f"[{self.id}] transitions from '{s.name}' under '{a}' "
                                             f"(matched via '{key}') sum to probability {total:g}, expected 1")
                        break
                if dist is None:
                    if self.strict_transitions:
                        raise ValueError(f"[{self.id}] no transition defined for state '{s.name}', action '{a}'")
                    dist = {s.name: 1.0}                          # default: stay
                table[(s.name, a)] = dist
        return table

    # ---- validation of user inputs
    def _validate(self):
        names = [s.name for s in self.states]
        if len(set(names)) != len(names):
            raise ValueError(f"[{self.id}] duplicate state names: {names}")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError(f"[{self.id}] duplicate actions: {self.actions}")
        for key, acts in self.available_actions.items():
            self._check_key(key, "available_actions")
            self._check_actions(acts, f"available_actions['{key}']")
        if isinstance(self.own_rewards, dict):
            for key, table in self.own_rewards.items():
                self._check_key(key, "own_rewards")
                self._check_actions(table, f"own_rewards['{key}']")
        for s in self.states:
            if not self.actions_in(s):
                raise ValueError(f"[{self.id}] state '{s.name}' has no available actions")
        for k, t in enumerate(self.transitions):
            missing = {"initial_state", "action", "final_state"} - set(t)
            if missing:
                raise ValueError(f"[{self.id}] transitions[{k}] is missing keys {sorted(missing)}")
            self._check_key(t["initial_state"], f"transitions[{k}].initial_state")
            self._check_actions([t["action"]], f"transitions[{k}].action")
            if t["final_state"] not in names:
                raise ValueError(f"[{self.id}] transitions[{k}].final_state '{t['final_state']}' "
                                 f"is not a state (phases are not allowed as final states)")
            p = t.get("probability", 1.0)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"[{self.id}] transitions[{k}].probability {p} not in [0, 1]")

    def _check_key(self, key, where):
        if not self.is_valid_key(key):
            raise ValueError(f"[{self.id}] {where}: '{key}' is not a state, phase or '*'")

    def _check_actions(self, acts, where):
        bad = [a for a in acts if a not in self.actions]
        if bad:
            raise ValueError(f"[{self.id}] {where}: unknown actions {bad}")


# =========================================================================== transfer rules
Amount = Union[Number, Callable[[Dict[str, State], Dict[str, str]], Number]]


@dataclass(eq=False)
class ResourceTransferRule:
    name: str
    when: Dict[str, Dict[str, Union[str, Sequence[str]]]]
    #   {stakeholder_id: {"state": key | [keys], "action": action | [actions]}}
    #   omitted "state"/"action" = any; all listed conditions must hold (AND)
    transfers: Dict[str, Amount]
    #   {stakeholder_id: reward change}; value may be f(states, actions) -> number
    predicate: Optional[Callable[[Dict[str, State], Dict[str, str]], bool]] = None
    #   optional extra condition for cases the declarative 'when' cannot express

    @classmethod
    def flow(cls, name, when, source, target, amount: Amount, predicate=None):
        """Conserved transfer: `source` gives `amount`, `target` receives it."""
        if callable(amount):
            transfers = {source: lambda s, a: -amount(s, a), target: amount}
        else:
            transfers = {source: -amount, target: amount}
        return cls(name, when, transfers, predicate)

    @property
    def scope(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys([*self.when, *self.transfers]))

    def _local(self, s, a):
        return {k: s[k] for k in self.scope}, {k: a[k] for k in self.scope}

    def fires(self, s: Dict[str, State], a: Dict[str, str]) -> bool:
        for sid, cond in self.when.items():
            if "state" in cond and not any(s[sid].matches(k) for k in _as_list(cond["state"])):
                return False
            if "action" in cond and a[sid] not in _as_list(cond["action"]):
                return False
        return self.predicate(*self._local(s, a)) if self.predicate else True

    def amounts(self, s, a) -> Dict[str, float]:
        s_loc, a_loc = self._local(s, a)
        return {sid: float(v(s_loc, a_loc) if callable(v) else v) for sid, v in self.transfers.items()}


# =========================================================================== constraints with memory
WAIT_ACTION = "wait (blocked)"      # only used if every action of a stakeholder is blocked (see MarkovGame)


def _keys(x) -> List[str]:
    return [] if x is None else [str(k) for k in _as_list(x)]


@dataclass
class VisitLimit:
    """How often `stakeholder` may visit `state` (a state name, a phase, or a list of them).

    count         "entries": a visit is an ENTRY into the state (coming from another state); staying
                             there does not count. For transitions that can come back (loops):
                             max_visits = 1 -> "once left, it cannot be visited again"
                  "steps"  : every time step spent in the state counts (a maximum duration)
    max_visits    the limit (0 = the state can never be entered)
    actions       count only visits caused by these actions (None = any action)
    count_initial the start state counts as the first visit, if it is the limited state
    when_reached  what happens to a move that would exceed the limit:
                  "mask": the action is removed from the stage game (with probabilistic transitions:
                          if any of its outcomes would exceed the limit)
                  "stay": the action stays available, but that move fails: its probability goes to
                          the current state (not possible if staying would itself exceed the limit,
                          e.g. count="steps" -> then the action is masked)
    """
    stakeholder: str
    state: Union[str, Sequence[str]]
    max_visits: int
    count: str = "entries"
    actions: Optional[Sequence[str]] = None
    count_initial: bool = True
    when_reached: str = "mask"
    name: Optional[str] = None


@dataclass
class ActionLimit:
    """How often `stakeholder` may take `action` (an action name or a list of them, counted together).

    max_times     the limit (0 = the action can never be taken there)
    state         count and limit the action only in these states/phases (None = in any state):
                  e.g. "stay" at most 2 times in "Planning; Design", or "use" at most 3 times in total
    Once the action has been taken max_times times, it is removed from the stage games ("mask") in the
    states where it counts; the other actions stay available.
    """
    stakeholder: str
    action: Union[str, Sequence[str]]
    max_times: int
    state: Optional[Union[str, Sequence[str]]] = None
    name: Optional[str] = None


@dataclass
class PrecedenceConstraint:
    """An action / a move of `stakeholder` is only valid once other stakeholders have been in given states.

    requires      {other stakeholder: state | phase | [states]}: that stakeholder must have been in one of
                  these states at some time up to now (being there now counts). Several stakeholders:
                  all of them are required. Checked at the start of the time step: a state reached in
                  this step unlocks the restriction from the next step on.
    actions       the restricted actions (None = any action)
    state         restrict only while `stakeholder` is in these states/phases (None = any state)
    to_state      restrict only the moves INTO these states/phases (None = any move the action makes);
                  moves within to_state (already there) are not restricted
    when_blocked  "mask": the restricted action is removed from the stage game while not allowed
                  "stay": the action stays available, but the restricted move fails (stays in place)
    At least one of `actions` and `to_state` must be given.
    """
    stakeholder: str
    requires: Dict[str, Union[str, Sequence[str]]]
    actions: Optional[Sequence[str]] = None
    state: Optional[Union[str, Sequence[str]]] = None
    to_state: Optional[Union[str, Sequence[str]]] = None
    when_blocked: str = "mask"
    name: Optional[str] = None


@dataclass(frozen=True)
class _Tracker:
    """One memory value a stakeholder keeps about its OWN history."""
    kind: str                                  # "entries" | "steps" | "actions" (counters) | "seen" (flag)
    keys: Tuple[str, ...]                      # state / phase keys
    actions: Optional[Tuple[str, ...]]         # counters: only visits caused by / times of these actions
    count_initial: bool
    label: str

    def hits(self, state: State) -> bool:
        return any(state.matches(k) for k in self.keys)

    def initial(self, state: State):
        if self.kind == "seen":
            return self.hits(state)
        if self.kind == "actions":
            return 0
        return 1 if (self.count_initial and self.hits(state)) else 0

    def update(self, value, state: State, action: str, nxt: State):
        if value is None:                      # no longer relevant
            return None
        if self.kind == "seen":
            return bool(value or self.hits(nxt))
        if self.actions is not None and action not in self.actions:
            return value
        if self.kind == "actions":                                   # the action taken in a counted state
            return value + 1 if self.hits(state) else value
        if self.kind == "entries":
            return value + 1 if (not self.hits(state) and self.hits(nxt)) else value
        return value + 1 if self.hits(nxt) else value          # "steps"

    def show(self, value) -> str:
        return ("yes" if value else "no") if self.kind == "seen" else str(value)


@dataclass(frozen=True)
class Node:
    """A stakeholder's state together with its memory (one value per tracker; None = no longer relevant).
    label = the name used in the stage-game keys."""
    state: State
    memory: Tuple
    label: str


# =========================================================================== stage game (output)
@dataclass
class StageGame:
    state: JointState                      # joint state, one state name per stakeholder
    index: Tuple[int, ...]                 # state indices -> the "i_j" in Q_game_i_j
    stakeholders: Tuple[str, ...]
    actions: List[List[str]]               # available actions per stakeholder, in index order
    R_self: np.ndarray                     # own rewards       [a_1, ..., a_n, i]
    R_transfer: np.ndarray                 # transferred       [a_1, ..., a_n, i]
    fired: Dict[Tuple[int, ...], List[str]]                       # joint action -> rules fired
    transitions: Dict[Tuple[int, ...], List[Tuple[JointState, float]]]
    #   joint action index -> [(next joint state, probability), ...]

    @property
    def name(self) -> str:
        return "_".join(map(str, self.index))

    @property
    def R(self) -> np.ndarray:             # total payoff      [a_1, ..., a_n, i]
        return self.R_self + self.R_transfer

    def payoff(self, stakeholder_id: str) -> np.ndarray:
        """Payoff tensor of one stakeholder (e.g. A, B matrices for a 2-player Nash solver)."""
        return self.R[..., self.stakeholders.index(stakeholder_id)]

    def as_list(self) -> List[np.ndarray]:
        """List indexed by the first stakeholder's action (2 players: rows = 2nd's action,
        columns = payoffs of each stakeholder)."""
        return list(self.R)

    def table(self) -> str:
        head = " | ".join(f"{sid}={st}" for sid, st in zip(self.stakeholders, self.state))
        lines = [f"Stage game {self.name}  [{head}]"]
        for idx in product(*[range(len(a)) for a in self.actions]):
            acts = ", ".join(self.actions[k][j] for k, j in enumerate(idx))
            pay = ", ".join(f"{v:g}" for v in self.R[idx])
            nxt = " + ".join(f"{', '.join(k)}" + (f" (p={p:g})" if p < 1 else "")
                             for k, p in self.transitions[idx])
            rules = f"   rules: {', '.join(self.fired[idx])}" if idx in self.fired else ""
            lines.append(f"  ({acts}) -> r = ({pay})  next: [{nxt}]{rules}")
        return "\n".join(lines)


# =========================================================================== Nash-Q result
@dataclass
class NashQResult:
    game: "MarkovGame"
    Q: Dict[JointState, np.ndarray]                       # learned Q tables [a_1, ..., a_n, agent]
    values: Dict[JointState, np.ndarray]                  # Nash value per agent of each stage game
    strategies: Dict[JointState, Optional[List[np.ndarray]]]
    residuals: List[float]                                # max |target - Q| per iteration
    iterations: int
    converged: bool
    history: Optional[dict] = None                        # recorded Q tables of tracked stage games (plots)

    def named(self, prefix: str = "Q_game") -> Dict[str, List[np.ndarray]]:
        """{'Q_game_0_1': list-of-arrays, ...} in the hand-written script format."""
        return {f"{prefix}_{self.game.index_name(k)}": list(q) for k, q in self.Q.items()}

    def policy_table(self) -> str:
        games = self.game.build()
        lines = []
        for k, strat in self.strategies.items():
            g = games[k]
            head = " | ".join(f"{sid}={st}" for sid, st in zip(g.stakeholders, k))
            vals = ", ".join(f"{v:.3f}" for v in self.values[k])
            lines.append(f"Game {g.name}  [{head}]   Nash value = ({vals})")
            if strat is None:
                lines.append("    (strategies not returned by the custom solver)")
                continue
            for sid, acts, p in zip(g.stakeholders, g.actions, strat):
                mix = ", ".join(f"{a}: {q:.2f}" for a, q in zip(acts, p) if q > 1e-9)
                lines.append(f"    {sid:<13} {mix}")
        return "\n".join(lines)


# =========================================================================== Monte Carlo simulation (output)
@dataclass
class Trajectory:
    """One simulated run of the Markov game over a fixed number of time steps."""
    stakeholders: Tuple[str, ...]
    states: List[JointState]               # T+1 joint states s_0 ... s_T
    games: List[str]                       # T+1 stage-game names ("i_j_k"), as in Q_game_i_j_k
    actions: List[Tuple[str, ...]]         # T joint actions, one action per stakeholder
    rewards: np.ndarray                    # (T, n) reward per step and stakeholder (own + transfers)
    rewards_self: np.ndarray               # (T, n) own-reward part
    rewards_transfer: np.ndarray           # (T, n) transferred part
    rules_fired: List[List[str]]           # T lists of transfer rules that fired
    action_probs: Optional[List[Tuple[float, ...]]] = None   # T: probability of each chosen action (policy runs)

    @property
    def cumulative(self) -> np.ndarray:    # (T, n) accumulated reward after each step
        return np.cumsum(self.rewards, axis=0)

    @property
    def total(self) -> np.ndarray:         # (n,) accumulated reward at the end of the run
        return self.rewards.sum(axis=0)

    def discounted_total(self, beta: float) -> np.ndarray:
        w = beta ** np.arange(len(self.rewards))
        return (w[:, None] * self.rewards).sum(axis=0)

    def table(self) -> str:
        fmt = lambda v: "(" + ", ".join(f"{x:g}" for x in np.round(v, 4)) + ")"
        cum = self.cumulative
        lines = ["Stakeholders: " + ", ".join(f"{i} = {sid}" for i, sid in enumerate(self.stakeholders))]
        for t, (js, a) in enumerate(zip(self.states, self.actions)):
            lines.append(f"t={t:<3} game {self.games[t]:<7} [" + " | ".join(js) + "]")
            rules = f"   rules: {', '.join(self.rules_fired[t])}" if self.rules_fired[t] else ""
            prob = (f"  prob ({', '.join(f'{p:.2f}' for p in self.action_probs[t])})"
                    if self.action_probs else "")
            lines.append(f"       actions ({', '.join(a)}){prob}  r = {fmt(self.rewards[t])}  "
                         f"cum = {fmt(cum[t])}{rules}")
        lines.append(f"t={len(self.actions):<3} game {self.games[-1]:<7} [" + " | ".join(self.states[-1]) + "]  (final)")
        return "\n".join(lines)


@dataclass
class MonteCarloResult:
    game: "MarkovGame"
    trajectories: List[Trajectory]
    n_steps: int

    # ---- arrays over all runs
    @property
    def rewards(self) -> np.ndarray:       # (runs, T, n)
        return np.stack([tr.rewards for tr in self.trajectories])

    @property
    def cumulative(self) -> np.ndarray:    # (runs, T, n) accumulated reward history
        return np.cumsum(self.rewards, axis=1)

    @property
    def totals(self) -> np.ndarray:        # (runs, n) accumulated reward at the final step
        return self.rewards.sum(axis=1)

    def cumulative_stats(self, percentiles=(5, 50, 95)) -> Dict[str, np.ndarray]:
        """Mean, std and percentiles of the accumulated reward at every step: each (T, n)."""
        c = self.cumulative
        out = {"mean": c.mean(axis=0), "std": c.std(axis=0)}
        for q in percentiles:
            out[f"p{q}"] = np.percentile(c, q, axis=0)
        return out

    def summary_by_step(self, percentiles=(5, 50, 95)) -> str:
        """Table of the accumulated reward after every time step, per stakeholder:
        mean, standard deviation and percentiles over all runs."""
        st = self.cumulative_stats(percentiles)
        cols = ["mean", "std"] + [f"p{q}" for q in percentiles]
        lines = [f"Accumulated reward per time step ({len(self.trajectories)} runs)"]
        for k, sid in enumerate(self.game.ids):
            lines.append(f"  {sid}")
            lines.append("    " + f"{'step':>5}" + "".join(f"{c:>10}" for c in cols))
            for t in range(self.n_steps):
                lines.append("    " + f"{t + 1:>5}" + "".join(f"{st[c][t, k]:>10.3f}" for c in cols))
        return "\n".join(lines)

    def save_cumulative_stats_csv(self, path: str, percentiles=(5, 50, 95)):
        """One row per time step and stakeholder: mean, std and percentiles of the accumulated reward."""
        import csv
        st = self.cumulative_stats(percentiles)
        cols = ["mean", "std"] + [f"p{q}" for q in percentiles]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "stakeholder"] + cols)
            for t in range(self.n_steps):
                for k, sid in enumerate(self.game.ids):
                    w.writerow([t + 1, sid] + [float(st[c][t, k]) for c in cols])

    # ---- state statistics
    def final_state_distribution(self) -> Dict[str, Dict[str, float]]:
        """Per stakeholder: fraction of runs ending in each state (memory ignored)."""
        out = {}
        for k, sid in enumerate(self.game.ids):
            names = [self.game.state_name(sid, tr.states[-1][k]) for tr in self.trajectories]
            out[sid] = {s.name: names.count(s.name) / len(names)
                        for s in self.game.stakeholders[sid].states if s.name in names}
        return out

    def state_occupancy(self, stakeholder_id: str) -> Tuple[List[str], np.ndarray]:
        """(state names, array (T+1, n_states)): fraction of runs in each state at each step."""
        k = self.game.ids.index(stakeholder_id)
        names = [s.name for s in self.game.stakeholders[stakeholder_id].states]
        occ = np.zeros((self.n_steps + 1, len(names)))
        for tr in self.trajectories:
            for t, js in enumerate(tr.states):
                occ[t, names.index(self.game.state_name(stakeholder_id, js[k]))] += 1
        return names, occ / len(self.trajectories)

    def most_common_game_sequences(self, top: int = 5) -> List[Tuple[Tuple[str, ...], int]]:
        counts: Dict[Tuple[str, ...], int] = {}
        for tr in self.trajectories:
            key = tuple(tr.games)
            counts[key] = counts.get(key, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])[:top]

    def summary(self, beta: Optional[float] = None) -> str:
        tot = self.totals
        lines = [f"Monte Carlo: {len(self.trajectories)} runs x {self.n_steps} steps",
                 f"{'stakeholder':<22}{'mean':>9}{'std':>9}{'p5':>9}{'p50':>9}{'p95':>9}   (accumulated reward)"]
        for k, sid in enumerate(self.game.ids):
            x = tot[:, k]
            lines.append(f"{sid:<22}{x.mean():>9.3f}{x.std():>9.3f}" +
                         "".join(f"{np.percentile(x, q):>9.3f}" for q in (5, 50, 95)))
        if beta is not None:
            disc = np.stack([tr.discounted_total(beta) for tr in self.trajectories])
            lines.append(f"discounted (beta = {beta}): " +
                         ", ".join(f"{sid} = {disc[:, k].mean():.3f}" for k, sid in enumerate(self.game.ids)))
        return "\n".join(lines)

    # ---- export
    def to_records(self) -> List[dict]:
        """Long format: one row per run and time step."""
        rows = []
        for r, tr in enumerate(self.trajectories):
            cum = tr.cumulative
            for t, a in enumerate(tr.actions):
                row = {"run": r, "t": t, "game": tr.games[t], "next_game": tr.games[t + 1]}
                for k, sid in enumerate(tr.stakeholders):
                    row[f"state | {sid}"] = tr.states[t][k]
                    row[f"action | {sid}"] = a[k]
                    row[f"reward | {sid}"] = float(tr.rewards[t, k])
                    row[f"cum_reward | {sid}"] = float(cum[t, k])
                row["rules_fired"] = "; ".join(tr.rules_fired[t])
                rows.append(row)
        return rows

    def to_dataframe(self):
        import pandas as pd                    # optional dependency
        return pd.DataFrame(self.to_records())

    def save_csv(self, path: str):
        import csv
        rows = self.to_records()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# =========================================================================== markov game
class MarkovGame:
    def __init__(self, stakeholders: Sequence[Stakeholder] = (), rules: Sequence[ResourceTransferRule] = (),
                 visit_limits: Sequence[VisitLimit] = (), precedence: Sequence[PrecedenceConstraint] = (),
                 when_no_action: str = "wait", action_limits: Sequence[ActionLimit] = ()):
        """when_no_action: what if constraints block EVERY action of a stakeholder in a stage game
            "wait" : it gets one action WAIT_ACTION ("wait (blocked)"): it stays where it is, memory unchanged,
                     own reward = its default_reward (transfer rules without an action condition still apply);
                     the stage games concerned are listed by blocked_stage_games() and in summary()
            "error": raise an error (add an always-allowed action, e.g. "stay", to that state instead)"""
        if when_no_action not in ("wait", "error"):
            raise ValueError("when_no_action must be 'wait' or 'error'")
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.rules: List[ResourceTransferRule] = []
        self.visit_limits: List[VisitLimit] = []
        self.action_limits: List[ActionLimit] = []
        self.precedence: List[PrecedenceConstraint] = []
        self.when_no_action = when_no_action
        self._index: Dict[str, List[ResourceTransferRule]] = {}
        self._cache: Optional[Dict[JointState, StageGame]] = None
        self._nodes: Optional[Dict[str, List[Node]]] = None
        self._blocked: List[JointState] = []
        for sh in stakeholders:
            self.add_stakeholder(sh)
        for r in rules:
            self.add_rule(r)
        for c in visit_limits:
            self.add_visit_limit(c)
        for c in action_limits:
            self.add_action_limit(c)
        for c in precedence:
            self.add_precedence(c)

    def _invalidate(self):
        self._cache, self._nodes = None, None

    # ---- inputs
    def add_stakeholder(self, sh: Stakeholder):
        if sh.id in self.stakeholders:
            raise ValueError(f"duplicate stakeholder id '{sh.id}'")
        self.stakeholders[sh.id] = sh
        self._invalidate()

    def add_rule(self, rule: ResourceTransferRule):
        for sid in rule.scope:
            if sid not in self.stakeholders:
                raise ValueError(f"rule '{rule.name}': unknown stakeholder '{sid}'")
        for sid, cond in rule.when.items():
            sh = self.stakeholders[sid]
            extra = set(cond) - {"state", "action"}
            if extra:
                raise ValueError(f"rule '{rule.name}': unknown condition keys {extra}")
            for key in _as_list(cond.get("state", [])):
                if not sh.is_valid_key(key):
                    raise ValueError(f"rule '{rule.name}': '{key}' is not a state/phase of '{sid}'")
            bad = [a for a in _as_list(cond.get("action", [])) if a not in sh.actions]
            if bad:
                raise ValueError(f"rule '{rule.name}': unknown actions {bad} for '{sid}'")
        self.rules.append(rule)
        for sid in rule.scope:
            self._index.setdefault(sid, []).append(rule)
        self._invalidate()

    def _check_keys(self, sid, keys, where):
        if sid not in self.stakeholders:
            raise ValueError(f"{where}: unknown stakeholder '{sid}'")
        for key in keys:
            if not self.stakeholders[sid].is_valid_key(key):
                raise ValueError(f"{where}: '{key}' is not a state/phase of '{sid}'")

    def _check_action_names(self, sid, acts, where):
        bad = [a for a in _keys(acts) if a not in self.stakeholders[sid].actions]
        if bad:
            raise ValueError(f"{where}: unknown actions {bad} for '{sid}'")

    def add_visit_limit(self, c: Union[VisitLimit, dict]):
        c = VisitLimit(**c) if isinstance(c, dict) else c
        where = f"visit limit '{c.name or c.state}'"
        self._check_keys(c.stakeholder, _keys(c.state), where)
        self._check_action_names(c.stakeholder, c.actions, where)
        if not _keys(c.state):
            raise ValueError(f"{where}: no state given")
        if c.count not in ("entries", "steps"):
            raise ValueError(f"{where}: count must be 'entries' or 'steps'")
        if c.when_reached not in ("mask", "stay"):
            raise ValueError(f"{where}: when_reached must be 'mask' or 'stay'")
        if int(c.max_visits) != c.max_visits or c.max_visits < 0:
            raise ValueError(f"{where}: max_visits must be a whole number >= 0")
        self.visit_limits.append(c)
        self._invalidate()

    def add_action_limit(self, c: Union[ActionLimit, dict]):
        c = ActionLimit(**c) if isinstance(c, dict) else c
        where = f"action limit '{c.name or c.action}'"
        acts = _keys(c.action)
        if not acts:
            raise ValueError(f"{where}: no action given")
        self._check_keys(c.stakeholder, _keys(c.state), where)
        self._check_action_names(c.stakeholder, acts, where)
        if int(c.max_times) != c.max_times or c.max_times < 0:
            raise ValueError(f"{where}: max_times must be a whole number >= 0")
        sh = self.stakeholders[c.stakeholder]
        keys = _keys(c.state) or [WILDCARD]
        if not any(any(st.matches(k) for k in keys) and set(acts) & set(sh.actions_in(st)) for st in sh.states):
            raise ValueError(f"{where}: {acts} is not available in any of the states {keys} of '{c.stakeholder}'")
        self.action_limits.append(c)
        self._invalidate()

    def add_precedence(self, c: Union[PrecedenceConstraint, dict]):
        c = PrecedenceConstraint(**c) if isinstance(c, dict) else c
        where = f"precedence constraint '{c.name or c.stakeholder}'"
        self._check_keys(c.stakeholder, _keys(c.state) + _keys(c.to_state), where)
        self._check_action_names(c.stakeholder, c.actions, where)
        if c.actions is None and c.to_state is None:
            raise ValueError(f"{where}: give the restricted 'actions' and/or 'to_state'")
        if not c.requires:
            raise ValueError(f"{where}: 'requires' is empty")
        for sid, keys in c.requires.items():
            self._check_keys(sid, _keys(keys), where)
            if not _keys(keys):
                raise ValueError(f"{where}: no required state for '{sid}'")
        if c.when_blocked not in ("mask", "stay"):
            raise ValueError(f"{where}: when_blocked must be 'mask' or 'stay'")
        self.precedence.append(c)
        self._invalidate()

    def rules_for(self, stakeholder_id: str) -> List[ResourceTransferRule]:
        """Rules that involve a given stakeholder (its interaction neighbourhood)."""
        return list(self._index.get(stakeholder_id, []))

    @classmethod
    def from_spec(cls, spec: dict) -> "MarkovGame":
        """Build from a plain dict (JSON-compatible unless callables are used)."""
        game = cls([Stakeholder(**sh) for sh in spec.get("stakeholders", [])])
        for r in spec.get("rules", []):
            if "flow" in r:
                f = r["flow"]
                game.add_rule(ResourceTransferRule.flow(r["name"], r.get("when", {}),
                                                        f["from"], f["to"], f["amount"],
                                                        r.get("predicate")))
            else:
                game.add_rule(ResourceTransferRule(r["name"], r.get("when", {}),
                                                   r["transfers"], r.get("predicate")))
        for c in spec.get("visit_limits", []):          # [{"stakeholder": ..., "state": ..., "max_visits": ...}]
            game.add_visit_limit(c)
        for c in spec.get("action_limits", []):         # [{"stakeholder": ..., "action": ..., "max_times": ...}]
            game.add_action_limit(c)
        for c in spec.get("precedence", []):            # [{"stakeholder": ..., "requires": {...}, "actions": ...}]
            game.add_precedence(c)
        if "when_no_action" in spec:
            game.when_no_action = spec["when_no_action"]
        return game

    # ---- structure
    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(self.stakeholders)

    # ---- memory (state augmentation) for the constraints
    def _ensure_nodes(self):
        """Per stakeholder: its trackers (memory values) and its nodes (state + memory) that can occur,
        found by following its own transitions from every state (as a possible start, empty history)."""
        if self._nodes is not None:
            return
        trackers: Dict[str, List[_Tracker]] = {sid: [] for sid in self.ids}

        def tracker_index(sid, tr):
            for i, t in enumerate(trackers[sid]):         # identical memory -> shared
                if (t.kind, t.keys, t.actions, t.count_initial) == (tr.kind, tr.keys, tr.actions, tr.count_initial):
                    return i
            trackers[sid].append(tr)
            return len(trackers[sid]) - 1

        self._limit_refs: Dict[str, List[Tuple[VisitLimit, int]]] = {}
        for c in self.visit_limits:
            keys = tuple(_keys(c.state))
            label = c.name or f"{'visits' if c.count == 'entries' else 'steps'} {'/'.join(keys)}"
            acts = tuple(c.actions) if c.actions is not None else None
            i = tracker_index(c.stakeholder, _Tracker(c.count, keys, acts, bool(c.count_initial), label))
            self._limit_refs.setdefault(c.stakeholder, []).append((c, i))
        self._action_refs: Dict[str, List[Tuple[ActionLimit, int]]] = {}
        for c in self.action_limits:
            keys = tuple(_keys(c.state)) or (WILDCARD,)
            acts = tuple(_keys(c.action))
            label = c.name or f"{'/'.join(acts)} times" + (f" in {'/'.join(keys)}" if c.state else "")
            i = tracker_index(c.stakeholder, _Tracker("actions", keys, acts, False, label))
            self._action_refs.setdefault(c.stakeholder, []).append((c, i))
        self._prec_refs: Dict[str, List[Tuple[PrecedenceConstraint, List[Tuple[str, int]]]]] = {}
        for c in self.precedence:
            reqs = []
            for sid, keys in c.requires.items():
                keys = tuple(_keys(keys))
                label = c.name or f"been in {'/'.join(keys)}"
                reqs.append((sid, tracker_index(sid, _Tracker("seen", keys, None, True, label))))
            self._prec_refs.setdefault(c.stakeholder, []).append((c, reqs))
        self._trackers = trackers

        # states reachable from each state (own transitions, any action): a counter whose state can no
        # longer be reached is no longer relevant and is dropped (None), so it does not multiply the states
        self._reach: Dict[str, Dict[str, set]] = {}
        for sid, sh in self.stakeholders.items():
            succ = {st.name: {s2 for a in sh.actions_in(st) for s2, p in sh.next_state_distribution(st, a).items()
                              if p > 0} for st in sh.states}
            reach = {}
            for st in sh.states:
                seen, stack = {st.name}, [st.name]
                while stack:
                    for s2 in succ[stack.pop()]:
                        if s2 not in seen:
                            seen.add(s2)
                            stack.append(s2)
                reach[st.name] = seen
            self._reach[sid] = reach

        self._node_store: Dict[str, Dict[Tuple[str, Tuple], Node]] = {sid: {} for sid in self.ids}
        self._initial: Dict[str, Dict[str, Node]] = {}
        nodes: Dict[str, List[Node]] = {}
        for sid, sh in self.stakeholders.items():
            starts = {st.name: self._make_node(sid, st, tuple(t.initial(st) for t in trackers[sid]))
                      for st in sh.states}
            self._initial[sid] = starts
            found, queue = set(starts.values()), list(starts.values())
            while queue:
                node = queue.pop()
                # precedence: enumerate both cases (requirements met / not met)
                for prec in ("met", "unmet"):
                    for _, dist in self._options(sid, node, prec):
                        for n2 in dist:
                            if n2 not in found:
                                found.add(n2)
                                queue.append(n2)
            order = {st.name: i for i, st in enumerate(sh.states)}
            key = lambda n: (order[n.state.name], tuple((v is None, v if v is not None else 0) for v in n.memory))
            nodes[sid] = [self._relabel(sid, n, found) for n in sorted(found, key=key)]
            self._initial[sid] = {name: self._node_store[sid][(n.state.name, n.memory)] for name, n in starts.items()}
        self._nodes = nodes
        self._by_label = {sid: {n.label: n for n in ns} for sid, ns in nodes.items()}
        self._pos = {sid: {n.label: i for i, n in enumerate(ns)} for sid, ns in nodes.items()}
        for sid, ns in nodes.items():
            if len(self._by_label[sid]) != len(ns):
                raise ValueError(f"[{sid}] two memory states got the same label: give the constraints names")

    def _still_counts(self, sid, t: _Tracker, reach) -> bool:
        """Can this counter still change? (its state, or for action counters: its action in its state,
        is still reachable). Counters that cannot are dropped (None) so they do not multiply the states."""
        sh = self.stakeholders[sid]
        for r in reach:
            st = sh.state(r)
            if t.hits(st) and (t.kind != "actions" or set(t.actions) & set(sh.actions_in(st))):
                return True
        return False

    def _make_node(self, sid, state: State, memory: Tuple) -> Node:
        trackers = self._trackers[sid]
        reach = self._reach[sid][state.name]
        memory = tuple(None if (v is not None and t.kind != "seen" and
                                not self._still_counts(sid, t, reach)) else v
                       for t, v in zip(trackers, memory))
        store = self._node_store[sid]
        if (state.name, memory) not in store:
            parts = [f"{t.label}={t.show(v)}" for t, v in zip(trackers, memory) if v is not None]
            label = state.name if not parts else f"{state.name} [{', '.join(parts)}]"
            store[(state.name, memory)] = Node(state, memory, label)
        return store[(state.name, memory)]

    def _relabel(self, sid, node: Node, found) -> Node:
        """Final label: the plain state name if that state occurs with one memory only; otherwise the
        state name + the memory values that differ between its stage-game states."""
        group = [n for n in found if n.state.name == node.state.name]
        trackers = self._trackers[sid]
        if len(group) == 1:
            label = node.state.name
        else:
            vary = [i for i in range(len(trackers)) if len({n.memory[i] for n in group}) > 1]
            label = f"{node.state.name} [{', '.join(f'{trackers[i].label}={trackers[i].show(node.memory[i])}' for i in vary)}]"
        new = Node(node.state, node.memory, label)
        self._node_store[sid][(node.state.name, node.memory)] = new
        return new

    def _advance(self, sid, node: Node, action: str, nxt: State) -> Node:
        memory = tuple(t.update(v, node.state, action, nxt) for t, v in zip(self._trackers[sid], node.memory))
        return self._make_node(sid, nxt, memory)

    def _options(self, sid, node: Node, others) -> List[Tuple[str, Dict[Node, float]]]:
        """Valid actions of one stakeholder in `node` and, for each, its next nodes {Node: probability}.
        others: {stakeholder: Node} of the stage game (for precedence constraints), or "met" / "unmet"
        (all requirements assumed met / not met; used to enumerate the nodes)."""
        sh = self.stakeholders[sid]
        st = node.state
        out = []
        used_up = {a for acts, _ in self._action_used_up(sid, node) for a in acts}   # action limits (own memory)
        for a in sh.actions_in(st):
            if a in used_up:                                            # taken max_times already: masked
                continue
            dist = {s2: p for s2, p in sh.next_state_distribution(st, a).items() if p > 0}
            ok = True
            for c, reqs in self._prec_refs.get(sid, []):                # precedence constraints
                if c.state is not None and not any(st.matches(k) for k in _keys(c.state)):
                    continue
                if c.actions is not None and a not in c.actions:
                    continue
                if others == "met" or (others != "unmet" and all(others[r].memory[i] for r, i in reqs)):
                    continue
                if c.to_state is None:
                    bad = [s2 for s2 in dist if s2 != st.name]
                    blocked = bool(bad) or c.when_blocked == "mask"
                elif any(st.matches(k) for k in _keys(c.to_state)):     # already inside: not an entry
                    continue
                else:
                    bad = [s2 for s2 in dist if any(sh.state(s2).matches(k) for k in _keys(c.to_state))]
                    blocked = bool(bad)
                if not blocked:
                    continue
                if c.when_blocked == "mask":
                    ok = False
                    break
                dist = self._redirect(dist, bad, st.name)
            if not ok:
                continue
            for c, i in self._limit_refs.get(sid, []):                  # visit limits (own memory)
                v, t = node.memory[i], self._trackers[sid][i]
                if v is None:
                    continue
                bad = [s2 for s2 in dist if t.update(v, st, a, sh.state(s2)) > c.max_visits]
                if not bad:
                    continue
                if c.when_reached == "mask" or st.name in bad:          # staying would exceed it too
                    ok = False
                    break
                dist = self._redirect(dist, bad, st.name)
            if not ok:
                continue
            nxt: Dict[Node, float] = {}
            for s2, p in dist.items():
                n2 = self._advance(sid, node, a, sh.state(s2))
                nxt[n2] = nxt.get(n2, 0.0) + p
            out.append((a, nxt))
        if not out:
            if self.when_no_action == "error" and not isinstance(others, str):
                raise ValueError(f"[{sid}] every action is blocked by the constraints in '{node.label}' "
                                 f"(stage game {self._label_of(others)}); add an always-allowed action "
                                 f"(e.g. 'stay') or use MarkovGame(when_no_action='wait')")
            out = [(WAIT_ACTION, {node: 1.0})]
        return out

    def _action_used_up(self, sid, node: Node) -> List[Tuple[Tuple[str, ...], int]]:
        """Action limits of this stakeholder that are reached in `node`: [(actions, index), ...]."""
        out = []
        for c, i in self._action_refs.get(sid, []):
            v, t = node.memory[i], self._trackers[sid][i]
            if v is not None and t.hits(node.state) and v >= c.max_times:
                out.append((t.actions, i))
        return out

    @staticmethod
    def _redirect(dist: Dict[str, float], bad, to: str) -> Dict[str, float]:
        """Moves the probability of the `bad` outcomes to state `to` (the move fails)."""
        out = {k: p for k, p in dist.items() if k not in bad}
        out[to] = out.get(to, 0.0) + sum(dist[k] for k in bad)
        return out

    def _label_of(self, others) -> str:
        return "(" + " | ".join(others[sid].label for sid in self.ids) + ")"

    def nodes(self, stakeholder_id: str) -> List[Node]:
        """States + memory of one stakeholder that occur in the stage games (= its states if no constraints)."""
        self._ensure_nodes()
        return list(self._nodes[stakeholder_id])

    def key(self, joint_state) -> JointState:
        """Stage-game key (tuple of labels) of a joint state. Each entry may be a label, a Node, a State
        or a plain state name; a plain name means that state with an empty history (as a start state),
        e.g. INITIAL_STATE = ("Policy Making", "Planning; Design", ...)."""
        self._ensure_nodes()
        joint_state = tuple(joint_state)
        if len(joint_state) != len(self.ids):
            raise ValueError(f"a joint state needs {len(self.ids)} entries ({', '.join(self.ids)})")
        out = []
        for sid, x in zip(self.ids, joint_state):
            if isinstance(x, Node):
                out.append(x.label)
                continue
            name = x.name if isinstance(x, State) else str(x)
            if name in self._by_label[sid]:
                out.append(name)
            elif name in self._initial[sid]:
                out.append(self._initial[sid][name].label)
            else:
                raise KeyError(f"[{sid}] unknown state '{name}'")
        return tuple(out)

    def _nodes_of(self, joint_state) -> Tuple[Node, ...]:
        k = self.key(joint_state)
        return tuple(self._by_label[sid][lab] for sid, lab in zip(self.ids, k))

    def state_name(self, stakeholder_id: str, label: str) -> str:
        """Plain state name of a label (memory removed)."""
        self._ensure_nodes()
        node = self._by_label[stakeholder_id].get(label)
        return node.state.name if node else self.stakeholders[stakeholder_id].state(label).name

    def joint_states(self) -> List[Tuple[Node, ...]]:
        """All joint (state + memory) combinations = all stage games."""
        self._ensure_nodes()
        return list(product(*[self._nodes[sid] for sid in self.ids]))

    def _resolve(self, joint_state) -> Tuple[State, ...]:
        return tuple(n.state for n in self._nodes_of(joint_state))

    def state_index(self, joint_state) -> Tuple[int, ...]:
        """Index of each stakeholder's (state + memory) -> the "i_j" in Q_game_i_j
        (without constraints: the state indices, as before)."""
        k = self.key(joint_state)
        return tuple(self._pos[sid][lab] for sid, lab in zip(self.ids, k))

    def index_name(self, joint_state) -> str:
        """'i_j' suffix of Q_game_i_j / r_game_i_j."""
        return "_".join(map(str, self.state_index(joint_state)))

    @staticmethod
    def _joint_next(dists: Sequence[Dict[Node, float]]) -> List[Tuple[JointState, float]]:
        out = []
        for combo in product(*[d.items() for d in dists]):
            p = prod(c[1] for c in combo)
            if p > 0:
                out.append((tuple(c[0].label for c in combo), p))
        return out

    def next_states(self, joint_state, joint_action: Dict[str, str]) -> List[Tuple[JointState, float]]:
        """Joint transition = product of each stakeholder's own transition (constraints applied)."""
        nodes = dict(zip(self.ids, self._nodes_of(joint_state)))
        dists = []
        for sid in self.ids:
            opts = dict(self._options(sid, nodes[sid], nodes))
            if joint_action[sid] not in opts:
                raise ValueError(f"[{sid}] action '{joint_action[sid]}' is not valid in '{nodes[sid].label}'")
            dists.append(opts[joint_action[sid]])
        return self._joint_next(dists)

    def _own_reward(self, sid: str, state: State, action: str) -> float:
        sh = self.stakeholders[sid]
        return float(sh.default_reward) if action == WAIT_ACTION else sh.own_reward(state, action)

    # ---- stage games
    def stage_game(self, joint_state) -> StageGame:
        key = self.key(joint_state)
        nodes = dict(zip(self.ids, self._nodes_of(key)))
        ids = self.ids
        s = {sid: nodes[sid].state for sid in ids}
        opts = [self._options(sid, nodes[sid], nodes) for sid in ids]     # constraints applied here
        acts = [[a for a, _ in o] for o in opts]
        shape = [len(x) for x in acts] + [len(ids)]
        R_self, R_tr, fired, trans = np.zeros(shape), np.zeros(shape), {}, {}

        for idx in product(*[range(len(x)) for x in acts]):
            a = {sid: acts[k][idx[k]] for k, sid in enumerate(ids)}
            R_self[idx] = [self._own_reward(sid, s[sid], a[sid]) for sid in ids]
            for rule in self.rules:
                if rule.fires(s, a):
                    for sid, v in rule.amounts(s, a).items():
                        R_tr[idx + (ids.index(sid),)] += v
                    fired.setdefault(idx, []).append(rule.name)
            trans[idx] = self._joint_next([opts[k][idx[k]][1] for k in range(len(ids))])

        return StageGame(key, self.state_index(key), ids, acts, R_self, R_tr, fired, trans)

    def build(self) -> Dict[JointState, StageGame]:
        """All stage games, keyed by joint state (tuple of labels: state name, plus memory if constrained)."""
        if self._cache is None:
            self._ensure_nodes()
            self._cache, self._blocked = {}, []
            for nodes in self.joint_states():
                g = self.stage_game(nodes)
                self._cache[g.state] = g
                if any(WAIT_ACTION in a for a in g.actions):
                    self._blocked.append(g.state)
            if self._blocked:
                warnings.warn(f"{len(self._blocked)} stage games have a stakeholder whose every action is blocked "
                              f"by the constraints; it waits there (see blocked_stage_games())", stacklevel=2)
        return self._cache

    def blocked_stage_games(self) -> List[JointState]:
        """Stage games in which a stakeholder had every action blocked (it got the WAIT_ACTION)."""
        self.build()
        return list(self._blocked)

    # ---- Monte Carlo simulation
    def _choose_actions(self, g: StageGame, t: int, policy, rng,
                        action_mode: str = "sample") -> Tuple[Tuple[int, ...], Optional[Tuple[float, ...]]]:
        """Joint action indices in stage game g (+ probability of each chosen action for policy runs).
        policy: "random"      -> each stakeholder picks uniformly among its available actions
                NashQResult   -> equilibrium strategy of each stage game (best policy):
                                 action_mode "sample" draws from it, "most_likely" takes its argmax
                callable      -> f(stage_game, t, rng) returning action indices or {stakeholder: action}"""
        if isinstance(policy, str):
            if policy != "random":
                raise ValueError(f"unknown policy '{policy}'")
            return tuple(int(rng.integers(len(a))) for a in g.actions), None
        if isinstance(policy, NashQResult):
            if g.state not in policy.strategies:
                raise KeyError(f"stage game {g.name} {g.state} was not solved in this NashQResult")
            strat = policy.strategies[g.state]
            if strat is None:
                raise ValueError("this NashQResult has no strategies: use a solver that returns them "
                                 "(e.g. NashEquilibriumStrategy)")
            idx, probs = [], []
            for p in strat:
                p = np.clip(np.asarray(p, dtype=float), 0.0, None)
                p = p / p.sum()
                j = int(np.argmax(p)) if action_mode == "most_likely" else int(rng.choice(len(p), p=p))
                idx.append(j)
                probs.append(float(p[j]))
            return tuple(idx), tuple(probs)
        if callable(policy):
            out = policy(g, t, rng)
            if isinstance(out, dict):
                return tuple(g.actions[k].index(out[sid]) for k, sid in enumerate(g.stakeholders)), None
            return tuple(int(i) for i in out), None
        raise TypeError("policy must be 'random', a NashQResult or a callable")

    def simulate(self, n_steps: int, initial_state=None, policy="random", rng=None, seed=None,
                 action_mode: str = "sample", transition_mode: str = "sample") -> Trajectory:
        """One run of n_steps: choose a joint action, collect rewards, move to the next joint state.
        action_mode     : "sample" | "most_likely"  (only for policy = NashQResult)
        transition_mode : "sample" -> draw the next game from the transition probabilities
                          "most_likely" -> always go to the most probable next game"""
        games = self.build()
        rng = rng if rng is not None else np.random.default_rng(seed)
        if initial_state is None:                              # default: first state of every stakeholder
            initial_state = tuple(sh.states[0].name for sh in self.stakeholders.values())
        state = self.key(initial_state)                        # plain names -> start with an empty history
        n = len(self.ids)
        states, names, actions, fired, probs = [state], [games[state].name], [], [], []
        R, R_self, R_tr = np.zeros((n_steps, n)), np.zeros((n_steps, n)), np.zeros((n_steps, n))

        for t in range(n_steps):
            g = games[state]
            idx, pr = self._choose_actions(g, t, policy, rng, action_mode)
            probs.append(pr)
            actions.append(tuple(g.actions[k][j] for k, j in enumerate(idx)))
            R[t], R_self[t], R_tr[t] = g.R[idx], g.R_self[idx], g.R_transfer[idx]
            fired.append(list(g.fired.get(idx, [])))
            outcomes = g.transitions[idx]                      # [(next joint state, probability), ...]
            if len(outcomes) == 1:
                state = outcomes[0][0]
            elif transition_mode == "most_likely":
                state = max(outcomes, key=lambda o: o[1])[0]
            else:
                p = np.array([q for _, q in outcomes])
                state = outcomes[int(rng.choice(len(outcomes), p=p / p.sum()))][0]
            states.append(state)
            names.append(games[state].name)

        action_probs = probs if all(p is not None for p in probs) else None
        return Trajectory(self.ids, states, names, actions, R, R_self, R_tr, fired, action_probs)

    def monte_carlo(self, n_runs: int, n_steps: int, initial_state=None, policy="random",
                    seed=None, action_mode: str = "sample", transition_mode: str = "sample") -> MonteCarloResult:
        """n_runs independent simulations of n_steps each (reproducible with seed).
        policy="random" for random actions, or a NashQResult to follow the best (equilibrium) policy."""
        rng = np.random.default_rng(seed)
        runs = [self.simulate(n_steps, initial_state, policy, rng=rng,
                              action_mode=action_mode, transition_mode=transition_mode)
                for _ in range(n_runs)]
        return MonteCarloResult(self, runs, n_steps)

    def best_policy_sequence(self, result: NashQResult, n_steps: int, initial_state=None) -> Trajectory:
        """The sequence of stage games and joint actions when every stakeholder follows the best
        (equilibrium) policy: most likely action in each game, most likely transition."""
        return self.simulate(n_steps, initial_state, policy=result,
                             action_mode="most_likely", transition_mode="most_likely")


    # ---- Nash Q-learning
    def solve_nash_q(self, beta: float = 0.9, n_iter: int = 1000, tol: float = 1e-6,
                     learning_rate: Union[str, Callable[[int], float]] = "harmonic",
                     nash_solver: Optional[Callable] = None, selection: Union[str, Callable] = "max_sum",
                     init: str = "zeros", sweep: str = "sync", verbose: bool = False,
                     n_jobs: int = 1, track_states: Optional[list] = None,
                     record_every: int = 1) -> NashQResult:
        """Full-sweep Nash-Q over every stage game and every joint action:
            Q_s[a] <- (1 - alpha_n) Q_s[a] + alpha_n (r_s[a] + beta * sum_{s'} P(s'|s,a) NashQ(s'))

        learning_rate : "harmonic" -> 1/(n+1) (as in the hand written script),
                        "constant" -> 1 (Nash value iteration), or a function n -> alpha
        nash_solver   : None -> built-in solver; or your function Q_game -> [value per agent]
                        (it receives the stage game as a list of arrays, like r_game_i_j).
                        NashEquilibriumMixedStrategy        -> same as the generated script
                        NashEquilibriumMixedStrategyProbs   -> same solver, also keeps the strategies
                                                               (needed for best_policy_sequence)
        n_jobs        : >1 computes the Nash values of the stage games of one iteration in parallel
                        worker processes (sweep="sync" only; identical results, just faster)
        sweep         : "sync"  -> Nash values of all games computed once per iteration, then all updated
                        "async" -> games updated in order; later games see earlier updates
        track_states  : stage games (joint states) whose Q tables and Nash values are recorded every
                        `record_every` iterations, for plot_q_iterations (does not change the results)
        """
        games = self.build()
        keys = list(games)
        lr = {"harmonic": lambda n: 1.0 / (n + 1), "constant": lambda n: 1.0}.get(learning_rate, learning_rate)
        Q = {k: games[k].R.copy() if init == "rewards" else np.zeros_like(games[k].R) for k in keys}

        def solve(q):
            if nash_solver is None:
                eq = nash_equilibrium(q, selection=selection)
                return eq.values, eq.strategies
            return _unpack(nash_solver(list(q)))

        def _unpack(out):
            if isinstance(out, tuple):                       # (Nash_values, probabilities)
                return np.asarray(out[0], dtype=float), [np.asarray(p, dtype=float) for p in out[1]]
            return np.asarray(out, dtype=float), None

        pool = None
        if n_jobs > 1 and nash_solver is not None and sweep == "sync":
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=n_jobs)

        track_states = [self.key(k) for k in (track_states or [])]
        history = {"iteration": [], "tracked_Q": {k: [] for k in track_states},
                   "tracked_nash": {k: [] for k in track_states}}

        def record(it):
            history["iteration"].append(it)
            for k in track_states:
                history["tracked_Q"][k].append(Q[k].copy())
                history["tracked_nash"][k].append(solve(Q[k])[0].copy())

        if track_states:
            record(0)                                        # initial Q tables

        residuals, converged, n = [], False, -1
        for n in range(n_iter):
            alpha = lr(n)
            cache: Dict[JointState, np.ndarray] = {}

            def nash_value(k):
                if k not in cache:
                    cache[k] = solve(Q[k])[0]
                return cache[k]

            if sweep == "sync":
                if pool:                                     # same Nash values, computed in parallel
                    outs = pool.map(nash_solver, [list(Q[k]) for k in keys],
                                    chunksize=max(1, len(keys) // (4 * n_jobs)))
                    for k, out in zip(keys, outs):
                        cache[k] = _unpack(out)[0]
                else:
                    for k in keys:
                        nash_value(k)
            res = 0.0
            for k in keys:
                g, q_new = games[k], Q[k].copy()
                for idx, outcomes in g.transitions.items():
                    target = g.R[idx] + beta * sum(p * nash_value(k2) for k2, p in outcomes)
                    res = max(res, float(np.abs(target - Q[k][idx]).max()))
                    q_new[idx] = (1 - alpha) * Q[k][idx] + alpha * target
                Q[k] = q_new
                if sweep == "async":
                    cache.pop(k, None)
            residuals.append(res)
            if verbose and (n % max(1, n_iter // 10) == 0):
                print(f"  iter {n:>6}  alpha = {alpha:.4f}  max|target - Q| = {res:.3e}")
            if track_states and ((n + 1) % record_every == 0 or res < tol or n == n_iter - 1):
                record(n + 1)
            if res < tol:
                converged = True
                break

        if pool:
            pool.shutdown()
        final = {k: solve(Q[k]) for k in keys}
        return NashQResult(self, Q, {k: v[0] for k, v in final.items()},
                           {k: v[1] for k, v in final.items()}, residuals, n + 1, converged,
                           history if track_states else None)

    def policy_from_q_tables(self, Q_tables: dict, nash_solver: Callable = None) -> NashQResult:
        """Best policy from Q-values computed elsewhere, e.g. by the generated script.
        Q_tables: {"Q_game_0_1": array, ...} or {"0_1": array, ...} or {joint state: array}.
        The strategies come from the same solver (default NashEquilibriumMixedStrategyProbs)."""
        nash_solver = nash_solver or NashEquilibriumMixedStrategyProbs
        by_name = {g.name: k for k, g in self.build().items()}
        Q, values, strategies = {}, {}, {}
        for key, q in Q_tables.items():
            if isinstance(key, str):
                k = by_name[key[len("Q_game_"):] if key.startswith("Q_game_") else key]
            else:
                k = self.key(key)
            Q[k] = np.asarray(q, dtype=float)
            vals, probs = nash_solver(list(Q[k]))
            values[k] = np.asarray(vals, dtype=float)
            strategies[k] = [np.asarray(p, dtype=float) for p in probs]
        return NashQResult(self, Q, values, strategies, [], 0, True)

    # ---- script generation (explicit, hand-written style)
    def generate_update_script(self, beta: float = 0.9, n_iter: int = 100,
                               solver_import: str = "import pygambit as gbt",
                               alpha_expr: str = "1 / (n + 1)") -> str:
        """Writes out every Nash-Q update explicitly (Q_game_i_j, r_game_i_j, NashQ_i_j_list),
        equivalent to solve_nash_q(sweep='sync') with the built-in or your own solver."""
        games = self.build()
        ids = self.ids
        L = ['"""Auto-generated Nash Q-learning update script."""',
             "import numpy as np",
             "import pygambit as gbt",
             "from SoS_Markov_game_with_transitions import NashEquilibriumMixedStrategy",
             "",
             f"beta = {beta}",
             f"N_ITER = {n_iter}",
             "",
             "# ---- reward arrays: r_game_<state indices>[a_0][a_1]...[agent]",
             f"# agents: " + ", ".join(f"{i} = {sid}" for i, sid in enumerate(ids))]
        for k, g in games.items():
            L.append(f"# {g.name}: " + ", ".join(f"{sid} = {st}" for sid, st in zip(ids, k)))
            L.append(f"r_game_{g.name} = np.array({g.R.tolist()})")
        L += ["", "# ---- Q tables"]
        L += [f"Q_game_{g.name} = np.zeros_like(r_game_{g.name})" for g in games.values()]
        L += ["", "for n in range(N_ITER):",
              f"    alpha = {alpha_expr}",
              "",
              "    # Nash values of every stage game from the current Q tables (each solved once per iteration)"]
        L += [f"    NashQ_{g.name}_list = NashEquilibriumMixedStrategy(list(Q_game_{g.name}))"
              for g in games.values()]
        for k, g in games.items():
            L += ["", f"    # --- Game {g.name}, {len(g.transitions)} action pairs, {len(ids)} agents ---",
                  f"    Q_game_{g.name}_new = Q_game_{g.name}.copy()"]
            for idx, outcomes in g.transitions.items():
                acts = ", ".join(g.actions[i][j] for i, j in enumerate(idx))
                nxt = " + ".join(f"{self.index_name(k2)}" + (f" (p={p:g})" if p < 1 else "") for k2, p in outcomes)
                L.append(f"    ## Action pair ({acts}) -> next game {nxt}")
                sub = "".join(f"[{j}]" for j in idx)
                for i, sid in enumerate(ids):
                    terms = [f"NashQ_{self.index_name(k2)}_list[{i}]" if p == 1
                             else f"{p:g} * NashQ_{self.index_name(k2)}_list[{i}]" for k2, p in outcomes]
                    future = terms[0] if len(terms) == 1 else "(" + " + ".join(terms) + ")"
                    L.append(f"    Q_game_{g.name}_new{sub}[{i}] = (1 - alpha) * Q_game_{g.name}{sub}[{i}]"
                             f" + alpha * (r_game_{g.name}{sub}[{i}] + beta * {future})   # agent {i}: {sid}")
            L.append(f"    Q_game_{g.name} = Q_game_{g.name}_new")
        L += ["", "if __name__ == '__main__':"]
        L += [f"    print('Q_game_{g.name} =', Q_game_{g.name}.tolist())" for g in games.values()]
        return "\n".join(L) + "\n"

    def save_update_script(self, path: str, **kwargs):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_update_script(**kwargs))

    # ---- outputs
    def reward_arrays(self) -> Dict[JointState, np.ndarray]:
        """Total payoff array R[a_1, ..., a_n, i] for every stage game."""
        return {k: g.R for k, g in self.build().items()}

    def named_reward_arrays(self, prefix: str = "r_game") -> Dict[str, List[np.ndarray]]:
        """{'r_game_0_1': list-of-arrays, ...} in the hand-written script format."""
        return {f"{prefix}_{g.name}": g.as_list() for g in self.build().values()}

    def to_dict(self) -> dict:
        def label(g, idx):
            return [g.actions[i][j] for i, j in enumerate(idx)]
        return {
            "stakeholders": list(self.ids),
            "stage_games": [
                {"name": g.name,
                 "state": dict(zip(self.ids, g.state)),
                 "actions": dict(zip(self.ids, g.actions)),
                 "R": g.R.tolist(),
                 "R_self": g.R_self.tolist(),
                 "R_transfer": g.R_transfer.tolist(),
                 "transitions": [{"joint_action": label(g, idx),
                                  "next": [{"state": list(k2), "game": self.index_name(k2), "p": p}
                                           for k2, p in out]}
                                 for idx, out in g.transitions.items()]}
                for g in self.build().values()
            ],
        }

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        games = self.build()
        dense = sum(g.R.size for g in games.values())
        lines = [f"Markov game: {len(self.ids)} stakeholders, {len(self.rules)} transfer rules, "
                 f"{len(games)} stage games, {dense} payoff entries generated"]
        for sh in self.stakeholders.values():
            lines.append(f"  {sh.id}: {len(sh.states)} states {[s.name for s in sh.states]}")
            lines.append(f"  {' ' * len(sh.id)}  phases: {sh.phases or '-'}, actions: {sh.actions}, "
                         f"{len(sh.transitions)} transition entries")
        if self.visit_limits or self.action_limits or self.precedence:
            lines.append(f"  constraints: {len(self.visit_limits)} visit limits, {len(self.action_limits)} action "
                         f"limits, {len(self.precedence)} precedence constraints")
            for c in self.visit_limits:
                what = "entries into" if c.count == "entries" else "time steps in"
                lines.append(f"    visit limit {c.name or ''}: {c.stakeholder} at most {c.max_visits} {what} "
                             f"{'/'.join(_keys(c.state))}"
                             + (f" (by {', '.join(c.actions)})" if c.actions else "") + f" [{c.when_reached}]")
            for c in self.action_limits:
                lines.append(f"    action limit {c.name or ''}: {c.stakeholder} at most {c.max_times} times "
                             f"{'/'.join(_keys(c.action))}"
                             + (f" in {'/'.join(_keys(c.state))}" if c.state else " (in any state)") + " [mask]")
            for c in self.precedence:
                what = ", ".join(filter(None, [f"actions {', '.join(c.actions)}" if c.actions else "",
                                               f"moves into {'/'.join(_keys(c.to_state))}" if c.to_state else ""]))
                where = f" in {'/'.join(_keys(c.state))}" if c.state else ""
                req = "; ".join(f"{sid} has been in {'/'.join(_keys(k))}" for sid, k in c.requires.items())
                lines.append(f"    precedence {c.name or ''}: {c.stakeholder} {what}{where} only after {req} "
                             f"[{c.when_blocked}]")
            for sid in self.ids:
                lines.append(f"    {sid}: {len(self._nodes[sid])} stage-game states (state + memory) "
                             f"from {len(self.stakeholders[sid].states)} states")
            if self._blocked:
                lines.append(f"    {len(self._blocked)} stage games with a fully blocked stakeholder (waits)")
        return "\n".join(lines)


# =========================================================================== plots: Q values over iterations
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
    """Direct labels at the right end of the lines. Lines ending at (almost) the same value share one
    label ("a2 / a4 / Nash"); labels are spread apart inside the panel so they don't overlap."""
    lo, hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    gap = 0.07 * (hi - lo)
    order = list(np.argsort(y_ends))
    groups = [[order[0]]]                             # lines whose ends are closer than half a gap
    for i in order[1:]:
        if y_ends[i] - y_ends[groups[-1][-1]] < 0.5 * gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    y_grp = np.array([np.mean([y_ends[i] for i in grp]) for grp in groups])
    pos = y_grp.copy()
    for i in range(1, len(pos)):                      # push up to keep the gap
        pos[i] = max(pos[i], pos[i - 1] + gap)
    if pos[-1] > hi:                                  # keep inside the panel: shift down from the top
        pos -= pos[-1] - hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    dx = 0.015 * (x_hi - x_lo)
    for grp, y0, y1 in zip(groups, y_grp, pos):
        text = " / ".join(labels[i] for i in sorted(grp))
        moved = abs(y1 - y0) > 1e-9 * max(1.0, abs(hi - lo))
        color = colors[grp[0]] if len(grp) == 1 else PLOT_STYLE["text2"]
        ax.annotate(text, xy=(x_end, y0), xytext=(x_end + dx, y1), textcoords="data",
                    va="center", fontsize=9, color=PLOT_STYLE["text"], annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8) if moved else None)


def plot_q_iterations(result: NashQResult, state, joint_actions=None, top: int = 4,
                      path: Optional[str] = None, show: bool = False):
    """Q values of one stage game over the iterations: one panel per stakeholder, one line per joint
    action, plus the Nash value of the stage game (dashed).
    joint_actions: list of joint actions as action names, e.g. [("fund", "manufacture", "develop 2")];
    default: the final equilibrium joint action, then the joint actions with the highest final total Q."""
    import matplotlib.pyplot as plt
    game = result.game
    state = game.key(state)
    if result.history is None or state not in result.history["tracked_Q"]:
        raise ValueError("state was not tracked: pass it in solve_nash_q(..., track_states=[...])")
    g = game.build()[state]
    hist = result.history
    x = np.array(hist["iteration"])
    snaps = np.stack(hist["tracked_Q"][state])                 # (records, a_1, ..., a_n, agents)
    nash_vals = np.stack(hist["tracked_nash"][state])         # (records, agents)

    if joint_actions is None:
        chosen = []
        strat = result.strategies.get(state)
        if strat is not None:                                  # final equilibrium joint action first
            chosen.append(tuple(int(np.argmax(p)) for p in strat))
        final_total = snaps[-1].sum(axis=-1)
        for idx in sorted(np.ndindex(*final_total.shape), key=lambda i: -final_total[i]):
            if len(chosen) >= top:
                break
            if idx not in chosen:
                chosen.append(idx)
        joint_actions = chosen
    else:
        joint_actions = [tuple(g.actions[k].index(nm) for k, nm in enumerate(ja)) if isinstance(ja[0], str)
                         else tuple(ja) for ja in joint_actions]
    joint_actions = joint_actions[:len(PLOT_STYLE["series"])]
    eq = tuple(int(np.argmax(p)) for p in result.strategies[state]) if result.strategies.get(state) else None
    names = [", ".join(g.actions[k][j] for k, j in enumerate(a)) + ("  [equilibrium]" if a == eq else "")
             for a in joint_actions]
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
        ax.set_xlabel("Iteration", fontsize=9, color=PLOT_STYLE["text2"])
        if i == 0:
            ax.set_ylabel("Q value", fontsize=9, color=PLOT_STYLE["text2"])
        ax.margins(x=0.02)

    handles = [plt.Line2D([], [], color=c, linewidth=2) for c in colors] + \
              [plt.Line2D([], [], color=PLOT_STYLE["nash"], linewidth=2, linestyle=(0, (4, 3)))]
    labels = [f"{code}: ({name})" for code, name in zip(codes, names)] + ["Nash value of the stage game"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=min(3, len(labels)),
               frameon=False, fontsize=9, labelcolor=PLOT_STYLE["text"])
    fig.suptitle(f"Q values of stage game {g.name} over the iterations   [{' | '.join(state)}]",
                 x=0.01, ha="left", fontsize=12, color=PLOT_STYLE["text"])
    fig.tight_layout(rect=(0, 0, 0.95, 0.84))
    if path:
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_convergence(result: NashQResult, tol: Optional[float] = None, path: Optional[str] = None,
                     show: bool = False):
    """Largest |target - Q| over all stage games per iteration (log scale)."""
    import matplotlib.pyplot as plt
    x = np.arange(1, len(result.residuals) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.patch.set_facecolor(PLOT_STYLE["surface"])
    _style_axis(ax)
    ax.plot(x, np.maximum(result.residuals, 1e-15), color=PLOT_STYLE["series"][0], linewidth=2)
    ax.set_yscale("log")
    if tol is not None:
        ax.axhline(tol, color=PLOT_STYLE["text2"], linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"tolerance {tol:g}", xy=(x[0], tol), xytext=(0, 4), textcoords="offset points",
                    fontsize=9, color=PLOT_STYLE["text2"])
    status = "converged" if result.converged else "not converged"
    ax.set_title(f"Convergence of the Q values ({status} after {result.iterations} iterations)",
                 fontsize=11, color=PLOT_STYLE["text"], loc="left")
    ax.set_xlabel("Iteration", fontsize=9, color=PLOT_STYLE["text2"])
    ax.set_ylabel("max |target - Q| over all stage games", fontsize=9, color=PLOT_STYLE["text2"])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# =========================================================================== demo helpers
def make_random_game(n_states=(3, 2), n_actions=(3, 2), n_rules=4,
                     reward_range=(-5, 5), seed=0) -> MarkovGame:
    """Synthetic game with customisable size (random own rewards, transfers, transitions)."""
    rng = np.random.default_rng(seed)
    lo, hi = reward_range
    stakeholders = []
    for k, (ns, na) in enumerate(zip(n_states, n_actions)):
        sid = f"S{k + 1}"
        states = [f"{sid}_s{j}" for j in range(ns)]
        actions = [f"{sid}_a{j}" for j in range(na)]
        own = {st: {a: int(rng.integers(lo, hi + 1)) for a in actions} for st in states}
        transitions = [{"initial_state": st, "action": a, "final_state": str(rng.choice(states))}
                       for st in states for a in actions]
        stakeholders.append(Stakeholder(sid, states, actions, own_rewards=own, transitions=transitions))
    game = MarkovGame(stakeholders)
    for r in range(n_rules):
        src, tgt = (str(x) for x in rng.choice(game.ids, 2, replace=False))
        when = {sid: {"state": str(rng.choice([s.name for s in game.stakeholders[sid].states])),
                      "action": str(rng.choice(game.stakeholders[sid].actions))}
                for sid in (src, tgt)}
        game.add_rule(ResourceTransferRule.flow(f"rule_{r}", when, src, tgt, int(rng.integers(1, hi + 1))))
    return game


# =========================================================================== demo
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)

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
                    "Acquisition Satellite #2; Manufacturing Satellite #1": {"stay":0, "operate 1":-0.1, "operate 1,2":-0.1+-0.2},
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
        ],

        # ---- constraints that depend on the history (handled with memory; see VisitLimit / PrecedenceConstraint)
        # 1. how often a stakeholder may visit a state
        #    count "entries": entries into the state (for transitions that can come back; max_visits=1 means
        #                     "once left, it cannot be visited again"). SoS1 has no transitions back to an
        #                     earlier state, so an "entries" limit would have no effect here, e.g.
        #                     {"stakeholder": "...", "state": "...", "max_visits": 1}
        #    count "steps"  : time steps spent in the state (a maximum duration), used below
        "visit_limits": [
            {"name": "policy steps", "stakeholder": "Ministry of Finance", "state": "Policy Making",
             "max_visits": 2, "count": "steps"},          # at most 2 time steps in Policy Making
        ],
        # 2. how often a stakeholder may take an action (optionally only in given states: "state": ...)
        "action_limits": [
            {"name": "data uses", "stakeholder": "Ministry of Finance", "action": "use",
             "max_times": 2},                              # the data may be used at most 3 times in total
            {"stakeholder": "Space agency", "action": "stay", "state": "Planning; Design", "max_times": 1}
        ],
        # 3. an action / a move of a stakeholder is only valid after another stakeholder has been in given states
        "precedence": [
            {"name": "sat 2 developed", "stakeholder": "Space agency",
             "to_state": "Operation Satellite #2; Operation Satellite #1",     # operating satellite #2 ...
             "requires": {"Private company": ["Low Battery, High Satellite #2", "High Battery, High Satellite #2",
                                              "High Battery, Operation Satellite #2"]}},   # ... needs it developed
            {"name": "satellite operating", "stakeholder": "Ministry of Finance",
             "state": "Funding #2", "actions": ["use"],                        # using the data ...
             "requires": {"Space agency": ["Acquisition Satellite #2; Operation Satellite #1",
                                           "Operation Satellite #2; Operation Satellite #1"]}},  # ... needs a satellite operating
        ],
    }
    # ------------ End of SoS1 use case ------------ #

    game = MarkovGame.from_spec(SoS1)
    games = game.build()
    print(game.summary(), "\n")

    # stage games with rewards and the next stage game for each action pair
    # for g in games.values():
    #     print(g.table(), "\n")

    # ---- settings
    BETA = 0.99
    INITIAL_STATE = ("Policy Making", "Planning; Design", "Low Battery, Low Satellite #2")
    N_STEPS = 8           # time steps per sequence
    N_RUNS = 1000          # Monte Carlo runs
    SEED = 1               # reproducibility
    N_JOBS = 1             # >1: Nash values of each iteration computed in parallel (same results)
    TRACK_STATES = [INITIAL_STATE,                                                    # stage games to plot
                    ("Funding #1", "Planning; Manufacturing Satellite #1", "Low Battery, High Satellite #2")]
    RECORD_EVERY = 1       # record the tracked Q tables every N iterations
    SHOW_PLOTS = True      # open the plot windows (they are also saved as PNG)

    # ---- Nash Q-values iteration
    N_ITER = 200
    res = game.solve_nash_q(beta=BETA, n_iter=N_ITER, tol=1e-6, learning_rate="constant", sweep="sync",
                            nash_solver=NashEquilibriumMixedStrategyProbs, n_jobs=N_JOBS, verbose=True,
                            track_states=TRACK_STATES, record_every=RECORD_EVERY)
    print(f"Nash-Q: {res.iterations} iterations, final max|target - Q| = {res.residuals[-1]:.3e}, "
          f"converged = {res.converged}\n")
    # print(res.policy_table())                  # equilibrium strategy of every stage game
    # Q_named = res.named()                      # {'Q_game_0_0_0': [...], ...}

    # Alternative: Q-values from the generated script itself, e.g. {"Q_game_0_0_0": Q_game_0_0_0, ...}
    # res = game.policy_from_q_tables(Q_tables_from_script)

    # ---- plots: Q values over the iterations
    for k in TRACK_STATES:
        name = games[game.key(k)].name
        plot_q_iterations(res, k, top=4, path=f"q_iterations_{name}.png")
        # or choose the joint actions yourself, e.g.
        # plot_q_iterations(res, k, joint_actions=[("fund", "manufacture", "develop 2")])
        print(f"Wrote q_iterations_{name}.png")
    plot_convergence(res, tol=1e-6, path="q_iterations_convergence.png")
    print("Wrote q_iterations_convergence.png\n")

    # ---- sequence of stage games and actions following the best (equilibrium) policy
    best = game.best_policy_sequence(res, n_steps=N_STEPS, initial_state=INITIAL_STATE)
    print("Best-policy sequence (most likely actions and transitions):")
    print(best.table(), "\n")

    # ---- best policy with sampled actions/transitions (spread of outcomes)
    mc_best = game.monte_carlo(n_runs=N_RUNS, n_steps=N_STEPS, initial_state=INITIAL_STATE,
                               policy=res, seed=SEED)
    print("Best policy, Monte Carlo:")
    print(mc_best.summary(beta=BETA))
    print(mc_best.summary_by_step())                 # mean, std, p5, p50, p95 after every time step
    mc_best.save_cumulative_stats_csv("sos1_best_policy_stats_by_step.csv")
    print("Wrote sos1_best_policy_stats_by_step.csv")
    for seq, count in mc_best.most_common_game_sequences(top=3):
        print(f"  {count:>4} runs: " + " -> ".join(seq))
    print()

    # ---- explicit script in the handwritten style, written automatically
    # (disabled: for SoS1 this file is ~3 MB, too large for PyCharm; use game.solve_nash_q instead)
    # game.save_update_script("generated_nash_q_update.py", beta=BETA, n_iter=50)
    # game.save_json("demo1_game.json")
    # print("Wrote demo1_game.json\n")

    # ---- Monte Carlo simulation: every stakeholder takes a random available action each step
    mc = game.monte_carlo(n_runs=N_RUNS, n_steps=N_STEPS, initial_state=INITIAL_STATE,
                          policy="random", seed=SEED)

    print("Random policy, example run (sequence of games, actions and rewards):")
    print(mc.trajectories[0].table(), "\n")

    print(mc.summary(beta=BETA), "\n")

    print("Final state distribution:")
    for sid, dist in mc.final_state_distribution().items():
        print(f"  {sid}: " + ", ".join(f"{s} {p:.1%}" for s, p in dist.items()))
    print()

    print("Most frequent sequences of stage games:")
    for seq, count in mc.most_common_game_sequences(top=3):
        print(f"  {count:>4} runs: " + " -> ".join(seq))
    print()

    stats = mc.cumulative_stats()      # mean / std / p5 / p50 / p95 of accumulated reward, each (T, n)
    print("Mean accumulated reward per step (rows = t, columns = stakeholders):")
    print(stats["mean"].round(2), "\n")
    print("Standard deviation of the accumulated reward per step (rows = t, columns = stakeholders):")
    print(stats["std"].round(2), "\n")
    print(mc.summary_by_step(), "\n")          # all statistics per stakeholder and time step
    mc.save_cumulative_stats_csv("sos1_random_policy_stats_by_step.csv")
    print("Wrote sos1_random_policy_stats_by_step.csv (mean, std, p5, p50, p95 per step and stakeholder)")

    reward_history = mc.rewards        # (runs, T, n) reward per step
    cum_history = mc.cumulative        # (runs, T, n) accumulated reward
    mc.save_csv("sos1_monte_carlo.csv")
    print("Wrote sos1_monte_carlo.csv (one row per run and time step)")

    if SHOW_PLOTS:
        import matplotlib.pyplot as plt
        plt.show()

    # ======================= Demo 2: customisable size ======================================
    # N_STATES = (4, 3)      # states per stakeholder
    # N_ACTIONS = (3, 2)     # actions per stakeholder
    # N_RULES = 6
    # REWARD_RANGE = (-5, 5)
    #
    # game2 = make_random_game(N_STATES, N_ACTIONS, N_RULES, REWARD_RANGE, seed=42)
    # print(game2.summary())
    # res2 = game2.solve_nash_q(beta=BETA, n_iter=2000, learning_rate="constant")
    # print(f"Nash-Q: {res2.iterations} iterations, converged = {res2.converged}")
    # first = next(iter(res2.Q))
    # print(f"Example Q table {first}: shape {res2.Q[first].shape}")