"""
Payoff-structure framework for Markov games (e.g. as input to Nash Q-learning).

    Stakeholder           states (optionally grouped under life-cycle phases), actions,
                          actions available per state, own (self) rewards
    ResourceTransferRule  conditional reward transfers between stakeholders,
                          triggered by their states and actions
    MarkovGame            holds stakeholders + rules, enumerates all stage games and
                          builds their reward arrays  R[a_1, ..., a_n, i]

Lookup keys used in user inputs (available actions, own rewards, rule conditions):
    a state name   e.g. "manufacture system 1"   (most specific)
    a phase name   e.g. "production"             (applies to every state in that phase)
    "*"            any state                     (least specific)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]
WILDCARD = "*"


# --------------------------------------------------------------------------- states
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


# --------------------------------------------------------------------------- stakeholder
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

    def __post_init__(self):
        self.states = _parse_states(self.states)
        self.actions = list(self.actions)
        self.available_actions = dict(self.available_actions or {})
        self.own_rewards = self.own_rewards if self.own_rewards is not None else {}
        self._validate()

    # ---- queries
    @property
    def phases(self) -> List[str]:
        return list(dict.fromkeys(s.phase for s in self.states if s.phase))

    def state(self, name: str) -> State:
        for s in self.states:
            if s.name == name:
                return s
        raise KeyError(f"[{self.id}] unknown state '{name}'")

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

    def _check_key(self, key, where):
        if not self.is_valid_key(key):
            raise ValueError(f"[{self.id}] {where}: '{key}' is not a state, phase or '*'")

    def _check_actions(self, acts, where):
        bad = [a for a in acts if a not in self.actions]
        if bad:
            raise ValueError(f"[{self.id}] {where}: unknown actions {bad}")


# --------------------------------------------------------------------------- transfer rules
Amount = Union[Number, Callable[[Dict[str, State], Dict[str, str]], Number]]


@dataclass
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


# --------------------------------------------------------------------------- stage game (output)
@dataclass
class StageGame:
    state: Tuple[str, ...]                 # joint state, one state name per stakeholder
    stakeholders: Tuple[str, ...]
    actions: List[List[str]]               # available actions per stakeholder, in index order
    R_self: np.ndarray                     # own rewards       [a_1, ..., a_n, i]
    R_transfer: np.ndarray                 # transferred       [a_1, ..., a_n, i]
    fired: Dict[Tuple[int, ...], List[str]]  # joint-action index -> names of rules that fired

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
        lines = [f"Stage game [{head}]"]
        for idx in product(*[range(len(a)) for a in self.actions]):
            acts = ", ".join(self.actions[k][j] for k, j in enumerate(idx))
            pay = ", ".join(f"{v:g}" for v in self.R[idx])
            rules = f"   <- {', '.join(self.fired[idx])}" if idx in self.fired else ""
            lines.append(f"  ({acts}) -> ({pay}){rules}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- markov game
class MarkovGame:
    def __init__(self, stakeholders: Sequence[Stakeholder] = (), rules: Sequence[ResourceTransferRule] = ()):
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.rules: List[ResourceTransferRule] = []
        self._index: Dict[str, List[ResourceTransferRule]] = {}
        self._cache: Optional[Dict[Tuple[str, ...], StageGame]] = None
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
        for sid in rule.scope:                       # index: stakeholder -> rules touching it
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

    # ---- computation
    def stage_game(self, joint_state) -> StageGame:
        js = self._resolve(joint_state)
        ids = self.ids
        s = dict(zip(ids, js))
        acts = [self.stakeholders[sid].actions_in(s[sid]) for sid in ids]
        shape = [len(x) for x in acts] + [len(ids)]
        R_self, R_tr, fired = np.zeros(shape), np.zeros(shape), {}
        for idx in product(*[range(len(x)) for x in acts]):
            a = {sid: acts[k][idx[k]] for k, sid in enumerate(ids)}
            R_self[idx] = [self.stakeholders[sid].own_reward(s[sid], a[sid]) for sid in ids]
            for rule in self.rules:
                if rule.fires(s, a):
                    for sid, v in rule.amounts(s, a).items():
                        R_tr[idx + (ids.index(sid),)] += v
                    fired.setdefault(idx, []).append(rule.name)

        return StageGame(tuple(x.name for x in js), ids, acts, R_self, R_tr, fired)

    def build(self) -> Dict[Tuple[str, ...], StageGame]:
        """All stage games, keyed by joint state (tuple of state names)."""
        if self._cache is None:
            self._cache = {}
            for js in self.joint_states():
                g = self.stage_game(js)
                self._cache[g.state] = g
        return self._cache

    # ---- outputs
    def reward_arrays(self) -> Dict[Tuple[str, ...], np.ndarray]:
        """Main output: total payoff array R[a_1, ..., a_n, i] for every stage game."""
        return {k: g.R for k, g in self.build().items()}

    def to_dict(self) -> dict:
        return {
            "stakeholders": list(self.ids),
            "stage_games": [
                {"state": dict(zip(self.ids, g.state)),
                 "actions": dict(zip(self.ids, g.actions)),
                 "R": g.R.tolist(),
                 "R_self": g.R_self.tolist(),
                 "R_transfer": g.R_transfer.tolist()}
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
            lines.append(f"  {sh.id}: {len(sh.states)} states "
                         f"(phases: {sh.phases or '-'}), {len(sh.actions)} actions {sh.actions}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- demo helpers
def make_random_game(n_states=(3, 2), n_actions=(3, 2), n_rules=4,
                     reward_range=(-5, 5), seed=0) -> MarkovGame:
    """Synthetic game with customisable size, for testing and scaling studies."""
    rng = np.random.default_rng(seed)
    lo, hi = reward_range
    stakeholders = []
    for k, (ns, na) in enumerate(zip(n_states, n_actions)):
        sid = f"S{k + 1}"
        states = [f"{sid}_s{j}" for j in range(ns)]
        actions = [f"{sid}_a{j}" for j in range(na)]
        own = {st: {a: int(rng.integers(lo, hi + 1)) for a in actions} for st in states}
        stakeholders.append(Stakeholder(sid, states, actions, own_rewards=own))
    game = MarkovGame(stakeholders)
    for r in range(n_rules):
        src, tgt = (str(x) for x in rng.choice(game.ids, 2, replace=False))
        when = {sid: {"state": str(rng.choice([s.name for s in game.stakeholders[sid].states])),
                      "action": str(rng.choice(game.stakeholders[sid].actions))}
                for sid in (src, tgt)}
        game.add_rule(ResourceTransferRule.flow(f"rule_{r}", when, src, tgt,
                                                int(rng.integers(1, hi + 1))))
    return game


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # ======================= Demo 1: SoS acquisition, 2 stakeholders (user inputs) ===========
    def acquisition_price(s, a):
        # price depends on which system the manufacturer is producing
        return 8 if s["Manufacturer"].name == "manufacture system 2" else 6

    spec = {
        "stakeholders": [
            {
                "id": "Government",
                "states": {                                   # phase -> states (hierarchy)
                    "development": ["low FoM_tech", "medium FoM_tech", "high FoM_tech"],
                    "operation": [],                          # phase used as a state itself
                },
                "actions": ["fund", "acquire", "wait"],
                "available_actions": {
                    "development": ["fund", "wait"],
                    "operation": ["acquire", "wait"],
                },
                "own_rewards": {
                    "*":            {"wait": -1},             # delay cost in any state
                    "high FoM_tech": {"fund": 2},             # funding a mature technology pays off
                    "operation":    {"acquire": 10},          # capability value of the SoS
                },
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
            {"name": "idle penalty",                          # non-conserved: plain transfers
             "when": {"Government": {"action": ["fund", "acquire"]},
                      "Manufacturer": {"action": "idle"}},
             "transfers": {"Government": -2, "Manufacturer": -1}},
        ],
    }

    game = MarkovGame.from_spec(spec)
    print(game.summary(), "\n")

    games = game.build()
    for key in [("high FoM_tech", "manufacture system 1"), ("operation", "manufacture system 2")]:
        print(games[key].table(), "\n")

    g = games[("operation", "manufacture system 2")]
    print("Output arrays for stage game", g.state)
    print("as list (outer = Government action, rows = Manufacturer action, cols = [Gov, Mfr]):")
    for k, arr in enumerate(g.as_list()):
        print(f"  Government '{g.actions[0][k]}':\n{arr}")
    print("Government payoff matrix:\n", g.payoff("Government"))
    print("Manufacturer payoff matrix:\n", g.payoff("Manufacturer"), "\n")

    game.save_json("demo1_reward_arrays.json")
    print("All reward arrays saved to demo1_reward_arrays.json\n")

    # ======================= Demo 2: customisable size ======================================
    N_STATES = (4, 3)      # states per stakeholder
    N_ACTIONS = (3, 2)     # actions per stakeholder
    N_RULES = 6
    REWARD_RANGE = (-5, 5)

    game2 = make_random_game(N_STATES, N_ACTIONS, N_RULES, REWARD_RANGE, seed=42)
    print(game2.summary())
    R_all = game2.reward_arrays()
    first = next(iter(R_all))
    print(f"\nExample: {first} -> array shape {R_all[first].shape}")
    print(game2.build()[first].table())