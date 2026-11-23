"""
SoS1 evaluation as a function:
    inputs : own-reward values and transfer-rule amounts (anything not given keeps its default)
    outputs: final cumulative reward and minimum cumulative reward of each stakeholder,
             along the best-policy sequence from Nash Q-value iteration

Needs SoS_Markov_game_with_transitions.py in the same folder.
"""
import copy

import numpy as np

# from SoS_markov_game_MonteCarlo_automated import MarkovGame, NashEquilibriumMixedStrategyProbs
from NashQ_iteration_MC_constrained import MarkovGame, NashEquilibriumMixedStrategyProbs

# ------------------------------------------------------------------ default SoS1 case
BASE_SOS1 = {
    "stakeholders": [
        {
            "id": "Ministry of Finance",
            "states": {"Policy Making": [], "Funding #1": [], "Funding #2": [], "Data Collection": [],
                       "End of Use": []},
            "actions": ["stay", "fund", "use", "end"],
            "available_actions": {
                "Policy Making": ["stay", "fund"],
                "Funding #1": ["stay", "fund"],
                "Funding #2": ["stay", "use"],
                "Data Collection": ["use", "end"],
                "End of Use": ["stay"],
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
            "states": {"Planning; Design": [], "Planning; Manufacturing Satellite #1": [],
                       "Acquisition Satellite #2; Manufacturing Satellite #1": [],
                       "Acquisition Satellite #2; Operation Satellite #1": [],
                       "Operation Satellite #2; Operation Satellite #1": [], "Retirement": []},
            "actions": ["stay", "manufacture", "acquire, manufacture", "acquire", "acquire, operate 1", "operate 1",
                        "operate 1,2", "retire"],
            "available_actions": {
                "Planning; Design": ["stay", "manufacture", "acquire, manufacture"],
                "Planning; Manufacturing Satellite #1": ["stay", "acquire", "acquire, operate 1"],
                "Acquisition Satellite #2; Manufacturing Satellite #1": ["stay", "operate 1", "operate 1,2"],
                "Acquisition Satellite #2; Operation Satellite #1": ["operate 1", "operate 1,2"],
                "Operation Satellite #2; Operation Satellite #1": ["operate 1,2", "retire"],
                "Retirement": ["stay"],
            },
            "own_rewards": {
                "Planning; Design": {"stay": -1, "manufacture": -1, "acquire, manufacture": -2},
                "Planning; Manufacturing Satellite #1": {"stay": -1, "acquire": 0, "acquire, operate 1": -0.1},
                "Acquisition Satellite #2; Manufacturing Satellite #1": {"stay": 0, "operate 1": -0.1,
                                                                         "operate 1,2": -0.1 + -0.2},
                "Acquisition Satellite #2; Operation Satellite #1": {"operate 1": -0.1, "operate 1,2": -0.1 - 0.2},
                "Operation Satellite #2; Operation Satellite #1": {"operate 1,2": -0.1 + -0.2, "retire": -1.2},
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
            "states": {"Low Battery, Low Satellite #2": [], "Low Battery, High Satellite #2": [],
                       "High Battery, High Satellite #2": [], "High Battery, Operation Satellite #2": [],
                       "End of Life Cycle": []},
            "actions": ["stay", "develop 2", "develop tech,2", "develop tech", "develop tech, operate 2",
                        "operate 2", "end"],
            "available_actions": {
                "Low Battery, Low Satellite #2": ["stay", "develop 2", "develop tech,2"],
                "Low Battery, High Satellite #2": ["stay", "develop tech", "develop tech, operate 2"],
                "High Battery, High Satellite #2": ["stay", "operate 2", "end"],
                "High Battery, Operation Satellite #2": ["operate 2", "end"],
                "End of Life Cycle": ["stay"],
            },
            "own_rewards": {
                "Low Battery, Low Satellite #2": {"stay": -1, "develop 2": -0.7, "develop tech,2": -0.3 - 0.7},
                "Low Battery, High Satellite #2": {"stay": -1, "develop tech": -0.3, "develop tech, operate 2": -0.3 - 0.4},
                "High Battery, High Satellite #2": {"stay": -1, "operate 2": -0.4, "end": -0.5},
                "High Battery, Operation Satellite #2": {"operate 2": -0.4, "end": -0.5},
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
        },
    ],
    "rules": [
        {"name": "system acquisition",
         "when": {"Space agency": {"action": ["acquire", "acquire, manufacture", "acquire, operate 1"]},
                  "Private company": {"state": "*"}},
         "flow": {"from": "Space agency", "to": "Private company", "amount": 3}},
        {"name": "technology acquisition",
         "when": {"Space agency": {"action": ["manufacture", "acquire, manufacture"]},
                  "Private company": {"state": "*"}},
         "flow": {"from": "Space agency", "to": "Private company", "amount": 1.5}},
        {"name": "funding #1",
         "when": {"Ministry of Finance": {"state": ["Policy Making", "Funding #1"], "action": "fund"},
                  "Space agency": {"state": "*"}},
         "flow": {"from": "Ministry of Finance", "to": "Space agency", "amount": 5}},
        {"name": "funding #2",
         "when": {"Ministry of Finance": {"state": "Funding #2", "action": "fund"},
                  "Space agency": {"state": "*"}},
         "flow": {"from": "Ministry of Finance", "to": "Space agency", "amount": 3}},
        {"name": "service transaction system 1",
         "when": {"Space agency": {"action": ["operate 1", "operate 1,2"]},
                  "Ministry of Finance": {"action": "use"}},
         "transfers": {"Space agency": 0.25, "Ministry of Finance": -0.3}},
        {"name": "service transaction system 2",
         "when": {"Space agency": {"action": "operate 1,2"},
                  "Private company": {"action": ["operate 2", "develop tech, operate 2"]},
                  "Ministry of Finance": {"action": "use"}},
         "transfers": {"Space agency": 0.15, "Private company": 0.1, "Ministry of Finance": -0.25}},
    ],
    "visit_limits": [
            {"name": "policy steps", "stakeholder": "Ministry of Finance", "state": "Policy Making",
             "max_visits": 2, "count": "steps"},          # at most 2 time steps in Policy Making
        ],
        # 2. an action / a move of a stakeholder is only valid after another stakeholder has been in given states
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

DEFAULT_INITIAL_STATE = ("Policy Making", "Planning; Design", "Low Battery, Low Satellite #2")


# ------------------------------------------------------------------ build the case with new values
def build_sos1(own_rewards=None, rule_amounts=None) -> dict:
    """SoS1 spec with some values replaced; everything not given keeps its default.

    own_rewards : {stakeholder: {state: {action: value}}}
                  e.g. {"Ministry of Finance": {"Data Collection": {"use": 3}}}
    rule_amounts: {rule name: amount} for "flow" rules (the amount moved from -> to),
                  {rule name: {stakeholder: value}} for "transfers" rules
                  e.g. {"funding #1": 6, "service transaction system 1": {"Space agency": 0.5}}
    """
    spec = copy.deepcopy(BASE_SOS1)
    by_id = {sh["id"]: sh for sh in spec["stakeholders"]}
    for sid, states in (own_rewards or {}).items():
        if sid not in by_id:
            raise KeyError(f"unknown stakeholder '{sid}'")
        sh = by_id[sid]
        for state, actions in states.items():
            if state not in sh["available_actions"]:
                raise KeyError(f"[{sid}] unknown state '{state}'")
            for action, value in actions.items():
                if action not in sh["available_actions"][state]:
                    raise KeyError(f"[{sid}] action '{action}' is not available in state '{state}'")
                sh["own_rewards"].setdefault(state, {})[action] = float(value)

    by_name = {r["name"]: r for r in spec["rules"]}
    for name, value in (rule_amounts or {}).items():
        if name not in by_name:
            raise KeyError(f"unknown rule '{name}' (rules: {list(by_name)})")
        rule = by_name[name]
        if "flow" in rule:
            rule["flow"]["amount"] = float(value)
        else:
            if not isinstance(value, dict):
                raise TypeError(f"rule '{name}' is a 'transfers' rule: give {{stakeholder: value}}")
            for sid, v in value.items():
                if sid not in rule["transfers"]:
                    raise KeyError(f"rule '{name}' has no transfer for '{sid}'")
                rule["transfers"][sid] = float(v)
    return spec


# ------------------------------------------------------------------ the function
def evaluate_sos1(own_rewards=None, rule_amounts=None, beta=0.99, n_iter=10, n_steps=8,
                  initial_state=DEFAULT_INITIAL_STATE, n_jobs=1, return_details=False):
    """Final and minimum cumulative reward of every stakeholder along the best-policy sequence.

    1. build SoS1 with the given own-reward values and rule amounts
    2. Nash Q-value iteration (alpha = 1, pygambit logit solver)
    3. best-policy sequence from initial_state over n_steps (most likely actions and transitions)

    Returns (final, minimum): dicts {stakeholder: value}
        final   : cumulative reward after the last time step
        minimum : lowest cumulative reward reached after any time step
    With return_details=True a third item is returned: {"trajectory", "result", "converged", "residual"}.
    """
    game = MarkovGame.from_spec(build_sos1(own_rewards, rule_amounts))
    res = game.solve_nash_q(beta=beta, n_iter=n_iter, tol=1e-6, learning_rate="constant", sweep="sync",
                            nash_solver=NashEquilibriumMixedStrategyProbs, n_jobs=n_jobs)
    traj = game.best_policy_sequence(res, n_steps=n_steps, initial_state=initial_state)

    cum = traj.cumulative                                   # (n_steps, stakeholders)
    final = {sid: float(cum[-1, k]) for k, sid in enumerate(game.ids)}
    minimum = {sid: float(cum[:, k].min()) for k, sid in enumerate(game.ids)}
    if return_details:
        return final, minimum, {"trajectory": traj, "result": res,
                                "converged": res.converged, "residual": res.residuals[-1]}
    return final, minimum


if __name__ == "__main__":
    # default values
    # final, minimum = evaluate_sos1()
    # print("final  :", final)
    # print("minimum:", minimum)

    # example with changed values
    final, minimum = evaluate_sos1(
        own_rewards={"Ministry of Finance": {"Data Collection": {"use": 3}}},
        rule_amounts={"funding #1": 6, "service transaction system 1": {"Space agency": 0.5}},
    )
    print("final  :", final)
    print("minimum:", minimum)