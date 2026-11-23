"""
Payoff + transition framework for Markov games, with an automated Nash Q-learning loop.

    Stakeholder           states (optionally grouped under life-cycle phases), actions,
                          actions available per state, own (self) rewards, state transitions
    ResourceTransferRule  conditional reward transfers between stakeholders,
                          triggered by their states and actions
    MarkovGame            holds stakeholders + rules, enumerates all stage games, builds
                          their reward arrays R[a_1, ..., a_n, i] and the joint transition
                          map (joint state, joint action) -> next stage game(s), and runs
                          Nash Q-learning over all stage games

Lookup keys used in user inputs (available actions, own rewards, transitions, rule conditions):
    a state name   e.g. "manufacture system 1"   (most specific)
    a phase name   e.g. "production"             (applies to every state in that phase)
    "*"            any state                     (least specific)

Naming convention for stage games (as in hand-written Nash-Q scripts):
    Q_game_i_j / r_game_i_j  ->  i = state index of stakeholder 0, j = state index of stakeholder 1, ...
"""
from __future__ import annotations

import json
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


# =========================================================================== Nash equilibrium solver
@dataclass
class Equilibrium:
    values: np.ndarray                     # expected payoff per agent
    strategies: List[np.ndarray]           # mixed strategy per agent (over its available actions)


def _pure_equilibria(Q: np.ndarray, tol: float) -> List[Equilibrium]:
    n, shape = Q.shape[-1], Q.shape[:-1]
    eqs = []
    for idx in np.ndindex(*shape):
        if all(Q[idx[:i] + (slice(None),) + idx[i + 1:] + (i,)].max() <= Q[idx + (i,)] + tol
               for i in range(n)):
            eqs.append(Equilibrium(Q[idx].copy(), [np.eye(shape[i])[idx[i]] for i in range(n)]))
    return eqs


def _indifference_mix(M: np.ndarray) -> Optional[np.ndarray]:
    """p >= 0, sum(p) = 1 such that every row of M @ p is equal; None if not solvable."""
    k, m = M.shape
    lhs = np.block([[M, -np.ones((k, 1))], [np.ones((1, m)), np.zeros((1, 1))]])
    rhs = np.zeros(k + 1)
    rhs[-1] = 1.0
    sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    if not np.allclose(lhs @ sol, rhs, atol=1e-9) or (sol[:-1] < -1e-9).any():
        return None
    p = np.clip(sol[:-1], 0.0, None)
    return p / p.sum()


def _check_2p(A, B, x, y, tol) -> Optional[Equilibrium]:
    u, w = A @ y, x @ B
    if u.max() > x @ u + tol or w.max() > w @ y + tol:
        return None
    return Equilibrium(np.array([x @ A @ y, x @ B @ y]), [x, y])


def _mixed_equilibria_2p(A, B, tol) -> List[Equilibrium]:
    """Support enumeration over equal-size supports (exact for non-degenerate games)."""
    m, n = A.shape
    eqs = []
    for k in range(2, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                y = _indifference_mix(A[np.ix_(I, J)])       # makes row player indifferent on I
                x = _indifference_mix(B[np.ix_(I, J)].T)     # makes column player indifferent on J
                if x is None or y is None:
                    continue
                xf, yf = np.zeros(m), np.zeros(n)
                xf[list(I)], yf[list(J)] = x, y
                eq = _check_2p(A, B, xf, yf, tol)
                if eq:
                    eqs.append(eq)
    return eqs


def _lp_equilibrium_2p(A, B, tol) -> List[Equilibrium]:
    """Fallback for degenerate games: LP feasibility over all support pairs (needs scipy)."""
    from scipy.optimize import linprog

    def mix(M, own, other):
        # find p on `other` (+ value u) so that rows `own` of M are best responses
        rows, k = M.shape[0], len(other)
        c = np.zeros(k + 1)
        A_eq = [np.r_[M[i, list(other)], -1.0] for i in own] + [np.r_[np.ones(k), 0.0]]
        b_eq = [0.0] * len(own) + [1.0]
        rest = [i for i in range(rows) if i not in own]
        A_ub = [np.r_[M[i, list(other)], -1.0] for i in rest] or None
        b_ub = [0.0] * len(rest) or None
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0, None)] * k + [(None, None)], method="highs")
        if not res.success:
            return None
        p = np.zeros(M.shape[1])
        p[list(other)] = res.x[:k]
        return p

    m, n = A.shape
    subsets = lambda r: [c for k in range(1, r + 1) for c in combinations(range(r), k)]
    for I in subsets(m):
        for J in subsets(n):
            y, x = mix(A, I, J), mix(B.T, J, I)
            if x is not None and y is not None:
                eq = _check_2p(A, B, x, y, tol)
                if eq:
                    return [eq]
    return []


SELECTION_RULES: Dict[str, Callable[[List[Equilibrium]], Equilibrium]] = {
    "max_sum": lambda eqs: max(eqs, key=lambda e: e.values.sum()),   # utilitarian
    "max_min": lambda eqs: max(eqs, key=lambda e: e.values.min()),   # egalitarian
    "first":   lambda eqs: eqs[0],
}


def nash_equilibrium(Q_game, selection: Union[str, Callable] = "max_sum",
                     prefer_pure: bool = True, tol: float = 1e-9) -> Equilibrium:
    """Nash equilibrium of one stage game. Q_game: ndarray or list-of-arrays, [a_1, ..., a_n, agent].
    2 players: pure + mixed (support enumeration, LP fallback). >2 players: pure only."""
    Q = np.asarray(Q_game, dtype=float)
    n = Q.shape[-1]
    eqs = _pure_equilibria(Q, tol)
    if n == 2 and (not eqs or not prefer_pure):
        A, B = Q[..., 0], Q[..., 1]
        eqs += _mixed_equilibria_2p(A, B, tol)
        if not eqs:
            eqs = _lp_equilibrium_2p(A, B, tol)
    if not eqs:
        hint = " (only pure equilibria are searched for >2 players: pass your own nash_solver)" if n > 2 else ""
        raise RuntimeError("no Nash equilibrium found" + hint)
    pick = selection if callable(selection) else SELECTION_RULES[selection]
    return pick(eqs)


# def NashEquilibriumMixedStrategy(Q_game) -> List[float]:
#     """Default solver with the same interface as a hand-written one: Nash value per agent."""
#     return nash_equilibrium(Q_game).values.tolist()


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
            lines.append(f"       actions ({', '.join(a)})  r = {fmt(self.rewards[t])}  "
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

    # ---- state statistics
    def final_state_distribution(self) -> Dict[str, Dict[str, float]]:
        """Per stakeholder: fraction of runs ending in each state."""
        out = {}
        for k, sid in enumerate(self.game.ids):
            names = [tr.states[-1][k] for tr in self.trajectories]
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
                occ[t, names.index(js[k])] += 1
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
    def __init__(self, stakeholders: Sequence[Stakeholder] = (), rules: Sequence[ResourceTransferRule] = ()):
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.rules: List[ResourceTransferRule] = []
        self._index: Dict[str, List[ResourceTransferRule]] = {}
        self._cache: Optional[Dict[JointState, StageGame]] = None
        for sh in stakeholders:
            self.add_stakeholder(sh)
        for r in rules:
            self.add_rule(r)

    # ---- inputs
    def add_stakeholder(self, sh: Stakeholder):
        if sh.id in self.stakeholders:
            raise ValueError(f"duplicate stakeholder id '{sh.id}'")
        self.stakeholders[sh.id] = sh
        self._cache = None

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
        self._cache = None

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
        return game

    # ---- structure
    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(self.stakeholders)

    def joint_states(self) -> List[Tuple[State, ...]]:
        return list(product(*[sh.states for sh in self.stakeholders.values()]))

    def _resolve(self, joint_state) -> Tuple[State, ...]:
        return tuple(st if isinstance(st, State) else self.stakeholders[sid].state(st)
                     for sid, st in zip(self.ids, joint_state))

    def state_index(self, joint_state) -> Tuple[int, ...]:
        return tuple(self.stakeholders[sid].state_index(st.name)
                     for sid, st in zip(self.ids, self._resolve(joint_state)))

    def index_name(self, joint_state) -> str:
        """'i_j' suffix of Q_game_i_j / r_game_i_j."""
        return "_".join(map(str, self.state_index(joint_state)))

    def next_states(self, joint_state, joint_action: Dict[str, str]) -> List[Tuple[JointState, float]]:
        """Joint transition = product of each stakeholder's own transition."""
        js = self._resolve(joint_state)
        dists = [self.stakeholders[sid].next_state_distribution(st, joint_action[sid]).items()
                 for sid, st in zip(self.ids, js)]
        out = []
        for combo in product(*dists):
            p = prod(c[1] for c in combo)
            if p > 0:
                out.append((tuple(c[0] for c in combo), p))
        return out

    # ---- stage games
    def stage_game(self, joint_state) -> StageGame:
        js = self._resolve(joint_state)
        ids = self.ids
        s = dict(zip(ids, js))
        acts = [self.stakeholders[sid].actions_in(s[sid]) for sid in ids]
        shape = [len(x) for x in acts] + [len(ids)]
        R_self, R_tr, fired, trans = np.zeros(shape), np.zeros(shape), {}, {}

        for idx in product(*[range(len(x)) for x in acts]):
            a = {sid: acts[k][idx[k]] for k, sid in enumerate(ids)}
            R_self[idx] = [self.stakeholders[sid].own_reward(s[sid], a[sid]) for sid in ids]
            for rule in self.rules:
                if rule.fires(s, a):
                    for sid, v in rule.amounts(s, a).items():
                        R_tr[idx + (ids.index(sid),)] += v
                    fired.setdefault(idx, []).append(rule.name)
            trans[idx] = self.next_states(js, a)

        return StageGame(tuple(x.name for x in js), self.state_index(js), ids, acts,
                         R_self, R_tr, fired, trans)

    def build(self) -> Dict[JointState, StageGame]:
        """All stage games, keyed by joint state (tuple of state names)."""
        if self._cache is None:
            self._cache = {}
            for js in self.joint_states():
                g = self.stage_game(js)
                self._cache[g.state] = g
        return self._cache

    # ---- Monte Carlo simulation
    def _choose_actions(self, g: StageGame, t: int, policy, rng) -> Tuple[int, ...]:
        """Joint action indices in stage game g.
        policy: "random"      -> each stakeholder picks uniformly among its available actions
                NashQResult   -> sample from the equilibrium (mixed) strategy of each stage game
                callable      -> f(stage_game, t, rng) returning action indices or {stakeholder: action}"""
        if isinstance(policy, str):
            if policy != "random":
                raise ValueError(f"unknown policy '{policy}'")
            return tuple(int(rng.integers(len(a))) for a in g.actions)
        if isinstance(policy, NashQResult):
            strat = policy.strategies[g.state]
            if strat is None:
                raise ValueError("this NashQResult has no strategies (custom solver): use the built-in solver")
            out = []
            for p in strat:
                p = np.clip(np.asarray(p, dtype=float), 0.0, None)
                out.append(int(rng.choice(len(p), p=p / p.sum())))
            return tuple(out)
        if callable(policy):
            out = policy(g, t, rng)
            if isinstance(out, dict):
                return tuple(g.actions[k].index(out[sid]) for k, sid in enumerate(g.stakeholders))
            return tuple(int(i) for i in out)
        raise TypeError("policy must be 'random', a NashQResult or a callable")

    def simulate(self, n_steps: int, initial_state=None, policy="random", rng=None, seed=None) -> Trajectory:
        """One run of n_steps: choose a joint action, collect rewards, sample the next joint state."""
        games = self.build()
        rng = rng if rng is not None else np.random.default_rng(seed)
        if initial_state is None:                              # default: first state of every stakeholder
            initial_state = tuple(sh.states[0].name for sh in self.stakeholders.values())
        state = tuple(st.name for st in self._resolve(initial_state))
        n = len(self.ids)
        states, names, actions, fired = [state], [games[state].name], [], []
        R, R_self, R_tr = np.zeros((n_steps, n)), np.zeros((n_steps, n)), np.zeros((n_steps, n))

        for t in range(n_steps):
            g = games[state]
            idx = self._choose_actions(g, t, policy, rng)
            actions.append(tuple(g.actions[k][j] for k, j in enumerate(idx)))
            R[t], R_self[t], R_tr[t] = g.R[idx], g.R_self[idx], g.R_transfer[idx]
            fired.append(list(g.fired.get(idx, [])))
            outcomes = g.transitions[idx]                      # [(next joint state, probability), ...]
            if len(outcomes) == 1:
                state = outcomes[0][0]
            else:
                p = np.array([pr for _, pr in outcomes])
                state = outcomes[int(rng.choice(len(outcomes), p=p / p.sum()))][0]
            states.append(state)
            names.append(games[state].name)

        return Trajectory(self.ids, states, names, actions, R, R_self, R_tr, fired)

    def monte_carlo(self, n_runs: int, n_steps: int, initial_state=None, policy="random",
                    seed=None) -> MonteCarloResult:
        """n_runs independent simulations of n_steps each (reproducible with seed)."""
        rng = np.random.default_rng(seed)
        runs = [self.simulate(n_steps, initial_state, policy, rng=rng) for _ in range(n_runs)]
        return MonteCarloResult(self, runs, n_steps)

    # ---- Nash Q-learning
    def solve_nash_q(self, beta: float = 0.9, n_iter: int = 1000, tol: float = 1e-6,
                     learning_rate: Union[str, Callable[[int], float]] = "harmonic",
                     nash_solver: Optional[Callable] = None, selection: Union[str, Callable] = "max_sum",
                     init: str = "zeros", sweep: str = "sync", verbose: bool = False) -> NashQResult:
        """Full-sweep Nash-Q over every stage game and every joint action:
            Q_s[a] <- (1 - alpha_n) Q_s[a] + alpha_n (r_s[a] + beta * sum_{s'} P(s'|s,a) NashQ(s'))

        learning_rate : "harmonic" -> 1/(n+1) (as in the hand-written script),
                        "constant" -> 1 (Nash value iteration), or a function n -> alpha
        nash_solver   : None -> built-in solver; or your function Q_game -> [value per agent]
                        (it receives the stage game as a list of arrays, like r_game_i_j)
        sweep         : "sync"  -> Nash values of all games computed once per iteration, then all updated
                        "async" -> games updated in order; later games see earlier updates
        """
        games = self.build()
        keys = list(games)
        lr = {"harmonic": lambda n: 1.0 / (n + 1), "constant": lambda n: 1.0}.get(learning_rate, learning_rate)
        Q = {k: games[k].R.copy() if init == "rewards" else np.zeros_like(games[k].R) for k in keys}

        def solve(q):
            if nash_solver is None:
                eq = nash_equilibrium(q, selection=selection)
                return eq.values, eq.strategies
            return np.asarray(nash_solver(list(q)), dtype=float), None

        residuals, converged, n = [], False, -1
        for n in range(n_iter):
            alpha = lr(n)
            cache: Dict[JointState, np.ndarray] = {}

            def nash_value(k):
                if k not in cache:
                    cache[k] = solve(Q[k])[0]
                return cache[k]

            if sweep == "sync":
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
            if res < tol:
                converged = True
                break

        final = {k: solve(Q[k]) for k in keys}
        return NashQResult(self, Q, {k: v[0] for k, v in final.items()},
                           {k: v[1] for k, v in final.items()}, residuals, n + 1, converged)

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
        return "\n".join(lines)


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

    # ======================= Demo 1: SoS acquisition, 2 stakeholders (user inputs) ===========
    def acquisition_price(s, a):
        # acquisition cost depends on which system's FoM level the private company is producing
        return 0 if s["Manufacturer"].name == ["Low Battery, Low Satellite #2"] else 0

    spec = {
        "stakeholders": [
            {
                "id": "Government",
                "states": {                                    # phase -> states (hierarchy)
                    "development": ["low FoM_tech", "medium FoM_tech", "high FoM_tech"],
                    "operation": [],                           # phase used as a state itself
                },
                "actions": ["fund", "acquire", "wait"],
                "available_actions": {
                    "development": ["fund", "wait"],
                    "operation": ["acquire", "wait"],
                },
                "own_rewards": {
                    "*":             {"wait": -1},             # delay cost in any state
                    "high FoM_tech": {"fund": 2},              # funding a mature technology pays off
                    "operation":     {"acquire": 10},          # capability value of the SoS
                },
                "transitions": [
                    {"initial_state": "low FoM_tech",    "action": "fund", "final_state": "medium FoM_tech"},
                    # stochastic maturation (optional "probability"; omitted = deterministic)
                    {"initial_state": "medium FoM_tech", "action": "fund", "final_state": "high FoM_tech",   "probability": 0.8},
                    {"initial_state": "medium FoM_tech", "action": "fund", "final_state": "medium FoM_tech", "probability": 0.2},
                    {"initial_state": "high FoM_tech",   "action": "fund", "final_state": "operation"},
                    # not listed (e.g. "wait", "acquire") -> stay in the same state
                ],
            },
            {
                "id": "Manufacturer",
                "states": {
                    "production": ["manufacture system 1", "manufacture system 2"],
                    "operation": ["support"],
                },
                "actions": ["invest", "manufacture", "maintain", "idle"],
                "available_actions": {
                    "production": ["invest", "manufacture", "idle"],
                    "operation": ["maintain", "idle"],
                },
                "own_rewards": {
                    "production": {"invest": -2, "manufacture": -4},
                    "manufacture system 2": {"manufacture": -5},   # overrides phase-level value
                    "operation": {"maintain": -1},
                },
                "transitions": [
                    {"initial_state": "manufacture system 1", "action": "invest",      "final_state": "manufacture system 2"},
                    {"initial_state": "production",           "action": "manufacture", "final_state": "support"},   # phase key
                    {"initial_state": "support",              "action": "idle",        "final_state": "manufacture system 1"},
                ],
            },
        ],
        "rules": [
            {"name": "R&D funding",
             "when": {"Government": {"state": "development", "action": "fund"},
                      "Manufacturer": {"action": "invest"}},
             "flow": {"from": "Government", "to": "Manufacturer", "amount": 3}},
            {"name": "acquisition payment",
             "when": {"Government": {"action": "acquire"},
                      "Manufacturer": {"state": "production", "action": "manufacture"}},
             "flow": {"from": "Government", "to": "Manufacturer", "amount": acquisition_price}},
            {"name": "sustainment contract",
             "when": {"Government": {"state": "operation"},
                      "Manufacturer": {"action": "maintain"}},
             "flow": {"from": "Government", "to": "Manufacturer", "amount": 2}},
            {"name": "idle penalty",                           # non-conserved: plain transfers
             "when": {"Government": {"action": ["fund", "acquire"]},
                      "Manufacturer": {"action": "idle"}},
             "transfers": {"Government": -2, "Manufacturer": -1}},
        ],
    }

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
                    "Funding #1": {"stay": 0, "fund": 0},
                    "Funding #2": {"stay": 0, "use": 2},
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
                    "Planning; Manufacturing Satellite #1": {"stay":0, "acquire":0, "acquire, operate 1":-0.1},
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
                    "Low Battery, High Satellite #2": {"stay":0, "develop tech":-0.3, "develop tech, operate 2":-0.3-0.4},
                    "High Battery, High Satellite #2": {"stay":0, "operate 2":-0.4, "end":-0.5},
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

    game = MarkovGame.from_spec(spec)
    games = game.build()
    print(game.summary(), "\n")

    # stage games with rewards and the next stage game for each action pair
    for g in games.values():
        print(g.table(), "\n")

    # ---- Nash Q-learning, automated over all stage games and action pairs
    BETA = 0.9
    # print("Nash-Q with alpha = 1/(n+1)  (as in the hand-written script):")
    # res_h = game.solve_nash_q(beta=BETA, n_iter=2000, tol=1e-6, learning_rate="harmonic")
    # print(f"  {res_h.iterations} iterations, final max|target - Q| = {res_h.residuals[-1]:.3e}, "
    #       f"converged = {res_h.converged}")
    #
    # print("Nash-Q with alpha = 1  (deterministic full sweeps = Nash value iteration):")
    # res_c = game.solve_nash_q(beta=BETA, n_iter=2000, tol=1e-6, learning_rate="constant")
    # print(f"  {res_c.iterations} iterations, final max|target - Q| = {res_c.residuals[-1]:.3e}, "
    #       f"converged = {res_c.converged}\n")
    #
    # print("Equilibrium policy per stage game (alpha = 1 run):")
    # print(res_c.policy_table(), "\n")
    #
    # Q_named = res_c.named()                    # {'Q_game_0_0': [...], ...}
    # r_named = game.named_reward_arrays()       # {'r_game_0_0': [...], ...}
    # print("Q_game_2_0 =", [q.round(3).tolist() for q in Q_named["Q_game_2_0"]], "\n")

    # ---- explicit script in the handwritten style, written automatically
    # (disabled: for SoS1 this file is ~3 MB, too large for PyCharm; use game.solve_nash_q instead)
    # game.save_update_script("generated_nash_q_update.py", beta=BETA, n_iter=50)
    game.save_json("demo1_game.json")
    print("Wrote demo1_game.json\n")

    # ---- Monte Carlo simulation: every stakeholder takes a random available action each step
    N_RUNS = 1000          # number of simulated life cycles
    N_STEPS = 7          # time steps per run
    SEED = 1               # for reproducibility
    # INITIAL_STATE = ("Policy Making", "Planning; Design", "Low Battery, Low Satellite #2")
    INITIAL_STATE = ("low FoM_tech", "manufacture system 1")

    mc = game.monte_carlo(n_runs=N_RUNS, n_steps=N_STEPS, initial_state=INITIAL_STATE,
                          policy="random", seed=SEED)

    print("Example random run (sequence of games, actions and rewards):")
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

    reward_history = mc.rewards        # (runs, T, n) reward per step
    cum_history = mc.cumulative        # (runs, T, n) accumulated reward
    mc.save_csv("sos1_monte_carlo.csv")
    print("Wrote sos1_monte_carlo.csv (one row per run and time step)")

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