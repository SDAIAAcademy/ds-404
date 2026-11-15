# %% [markdown]
# # Lab 3.2 · Orders and trucks for one depot week
#
# Day 3 · Session 2 (Optimising under capacity) · 25 min.
#
# The newsvendor quantities of Lab 3.1 ignore that products share trucks, that chilled dairy needs one of two
# reefers, and that the Riyadh depot can receive only 200 cubic metres a week. You write the week as an
# integer programme (whole cases, whole trucks), solve it with HiGHS through PuLP, read the plan and find which
# constraint binds and what one more unit of it is worth.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `U.week_inputs()` (the 20 products, their costs and newsvendor targets for 11 to
# 17 May 2026) and `data/trucks.json` (synthetic rates).
#
# **Tasks**
# - [ ] 1 · Variables and objective (6')
# - [ ] 2 · Constraints: trucks, reefers, space (7')
# - [ ] 3 · Solve and read the plan (6')
# - [ ] 4 · Which constraint binds (6')
# - [ ] ✱ Stretch · Add a minimum order per supplier

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-404.git"
    if not Path("/content/ds-404").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-404"], check=True)
    os.chdir("/content/ds-404/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import pulp
import bustan as U

pd.set_option("display.width", 160)
p = U.week_inputs()
T = U.js("trucks.json")
print(T["note"])
vol = (p.target * p.case_m3).sum()
print(f"newsvendor targets need {vol:.1f} m3; the depot receives {T['depot_inbound_m3_per_week']:.0f} m3 a week")

# %% [markdown]
# ## 1 · Variables and objective (6')
#
# For each product *i*: `q[i]` whole cases ordered, `short[i]` and `over[i]` cases below and above the
# newsvendor target (`q[i] - target[i] = over[i] - short[i]`). For each supplier *s*: `trucks[s]` whole
# trucks (reefers for S3, the chilled supplier).
#
# Objective: truck trips at their cost, plus `underage * short + overage * over`. This charges each case away
# from the target at the newsvendor's own rates: a linear stand-in for the expected cost, good enough to share
# scarce capacity. (Lab 3.4's stretch solves the exact version on scenarios.)
#
# ✏️ **Exercise** · Create the problem, the variables and the objective. Return `(prob, q, short, over, trucks)`.

# %%
SUPPLIERS = sorted(p.supplier.unique())


def build(p: pd.DataFrame, T: dict):
    # BEGIN SOLUTION
    prob = pulp.LpProblem("bustan_week", pulp.LpMinimize)
    q = {i: prob.add_variable(f"q_{r['product']}", lowBound=0, cat="Integer") for i, r in p.iterrows()}
    short = {i: prob.add_variable(f"short_{r['product']}", lowBound=0) for i, r in p.iterrows()}
    over = {i: prob.add_variable(f"over_{r['product']}", lowBound=0) for i, r in p.iterrows()}
    trucks = {s: prob.add_variable(f"trucks_{s}", lowBound=0, cat="Integer") for s in SUPPLIERS}
    cost = {s: T["reefer" if s == "S3" else "truck"]["sar_per_trip"] for s in SUPPLIERS}
    prob += (pulp.lpSum(cost[s] * trucks[s] for s in SUPPLIERS)
             + pulp.lpSum(r.underage * short[i] + r.overage * over[i] for i, r in p.iterrows()))
    for i, r in p.iterrows():
        prob += q[i] - r.target == over[i] - short[i], f"target_{r['product']}"
    return prob, q, short, over, trucks
    # END SOLUTION

# %%
def ref_build(p, T):
    prob = pulp.LpProblem("bustan_week", pulp.LpMinimize)
    q = {i: prob.add_variable(f"q_{r['product']}", lowBound=0, cat="Integer") for i, r in p.iterrows()}
    short = {i: prob.add_variable(f"short_{r['product']}", lowBound=0) for i, r in p.iterrows()}
    over = {i: prob.add_variable(f"over_{r['product']}", lowBound=0) for i, r in p.iterrows()}
    trucks = {s: prob.add_variable(f"trucks_{s}", lowBound=0, cat="Integer") for s in SUPPLIERS}
    cost = {s: T["reefer" if s == "S3" else "truck"]["sar_per_trip"] for s in SUPPLIERS}
    prob += (pulp.lpSum(cost[s] * trucks[s] for s in SUPPLIERS)
             + pulp.lpSum(r.underage * short[i] + r.overage * over[i] for i, r in p.iterrows()))
    for i, r in p.iterrows():
        prob += q[i] - r.target == over[i] - short[i], f"target_{r['product']}"
    return prob, q, short, over, trucks


try:
    prob, q, short, over, trucks = build(p, T)
except NotImplementedError:
    build = ref_build
    prob, q, short, over, trucks = build(p, T)
U.check(len(q) == 20 and len(trucks) == 4, "20 order variables and 4 truck variables")

# %% [markdown]
# ## 2 · Constraints: trucks, reefers, space (7')
#
# - Each supplier's cubic metres fit in its trucks: `sum(case_m3 * q) <= m3_per_truck * trucks[s]`; the same
#   for kilograms.
# - S3 ships only in reefers, and at most `available_per_week` of them.
# - All 20 products together fit the depot's weekly inbound space.
#
# ✏️ **Exercise** · Add the constraints to `prob`, named `m3_<supplier>`, `kg_<supplier>`, `reefers` and
# `depot_space`.

# %%
def add_constraints(prob, p, T, q, trucks, depot_m3=None):
    # BEGIN SOLUTION
    for s in SUPPLIERS:
        kind = T["reefer" if s == "S3" else "truck"]
        rows = p[p.supplier == s]
        prob += pulp.lpSum(r.case_m3 * q[i] for i, r in rows.iterrows()) <= kind["m3"] * trucks[s], f"m3_{s}"
        prob += pulp.lpSum(r.case_kg * q[i] for i, r in rows.iterrows()) <= kind["kg"] * trucks[s], f"kg_{s}"
    prob += trucks["S3"] <= T["reefer"]["available_per_week"], "reefers"
    space = T["depot_inbound_m3_per_week"] if depot_m3 is None else depot_m3
    prob += pulp.lpSum(r.case_m3 * q[i] for i, r in p.iterrows()) <= space, "depot_space"
    return prob
    # END SOLUTION

# %%
def ref_add_constraints(prob, p, T, q, trucks, depot_m3=None):
    for s in SUPPLIERS:
        kind = T["reefer" if s == "S3" else "truck"]; rows = p[p.supplier == s]
        prob += pulp.lpSum(r.case_m3 * q[i] for i, r in rows.iterrows()) <= kind["m3"] * trucks[s], f"m3_{s}"
        prob += pulp.lpSum(r.case_kg * q[i] for i, r in rows.iterrows()) <= kind["kg"] * trucks[s], f"kg_{s}"
    prob += trucks["S3"] <= T["reefer"]["available_per_week"], "reefers"
    space = T["depot_inbound_m3_per_week"] if depot_m3 is None else depot_m3
    prob += pulp.lpSum(r.case_m3 * q[i] for i, r in p.iterrows()) <= space, "depot_space"
    return prob


try:
    prob = add_constraints(prob, p, T, q, trucks)
except NotImplementedError:
    add_constraints = ref_add_constraints
    prob = add_constraints(prob, p, T, q, trucks)
U.check({"depot_space", "reefers", "m3_S1", "kg_S2"} <= {c.name for c in prob.constraints()}, "the named constraints are in the problem")

# %% [markdown]
# ## 3 · Solve and read the plan (6')

# %%
stats = prob.solve(pulp.HiGHS(msg=False))             # PuLP 4 returns the solve statistics
U.check(stats.status == pulp.LpSolveStatus.Optimal, "HiGHS finds an optimal plan")
plan = p[["product", "category", "supplier", "cr", "target"]].copy()
plan["order"] = [round(q[i].value()) for i in p.index]
plan["cut"] = plan.target - plan.order
print(plan.to_string(index=False))
tr = {s: round(trucks[s].value()) for s in SUPPLIERS}
print("trucks:", tr, "| objective SAR", round(pulp.value(prob.objective)))
used = float((plan.order * p.case_m3).sum())
print(f"inbound volume {used:.1f} of {T['depot_inbound_m3_per_week']:.0f} m3")

# %% [markdown]
# **Look at.** Which products were cut below their target, and which supplier's truck is full? At their targets
# the dry goods need 75 cubic metres and 25 tonnes: two trucks. Cut by about 15 cubic metres they fit one, which
# saves SAR 2,400 and also frees the depot space that binds. The programme takes the cut where a missing case
# costs least per kilogram and cubic metre of that truck.

# %% [markdown]
# ## 4 · Which constraint binds (6')
#
# A constraint **binds** when it holds with no slack. What is one more unit worth? Re-solve with 10 more
# cubic metres of depot space and compare the objective.
#
# ✏️ **Exercise** · Return a table of the constraints that start with `m3_`, `kg_`, `reefers` and
# `depot_space`, with their slack (`constraint.slack`, rounded to 2 decimals) and a column `binds`.

# %%
def slack_table(prob) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = [{"constraint": c.name, "slack": round(c.slack, 2)} for c in prob.constraints()
            if c.name.startswith(("m3_", "kg_", "reefers", "depot_space"))]
    t = pd.DataFrame(rows)
    return t.assign(binds=t.slack.abs() < 0.5)
    # END SOLUTION

# %%
try:
    slack = slack_table(prob)
except NotImplementedError:
    rows = [{"constraint": c.name, "slack": round(c.slack, 2)} for c in prob.constraints() if c.name.startswith(("m3_", "kg_", "reefers", "depot_space"))]
    slack = pd.DataFrame(rows); slack["binds"] = slack.slack.abs() < 0.5
print(slack.to_string(index=False))
prob2, q2, s2, o2, t2 = build(p, T)
prob2 = add_constraints(prob2, p, T, q2, t2, depot_m3=T["depot_inbound_m3_per_week"] + 10)
prob2.solve(pulp.HiGHS(msg=False))
value10 = pulp.value(prob.objective) - pulp.value(prob2.objective)
print(f"10 more m3 of depot space would save SAR {value10:,.0f} this week")
U.check(slack.set_index("constraint").loc["depot_space", "binds"], "the depot's inbound space binds")
U.save_reference("d3_2", {"plan": plan.to_dict("records"), "trucks": tr, "objective": pulp.value(prob.objective),
                          "used_m3": used, "slack": slack.to_dict("records"), "value_10m3": value10})

# %% [markdown]
# ## ✱ Stretch · A minimum order per supplier
#
# S4 delivers only orders of at least 5 cubic metres. Model it with a binary "order from S4 this week"
# variable linked to its trucks. What does the plan do with the dates?

# %% [markdown]
# ## Checkpoint
#
# 1. Why are trucks integer variables, not continuous ones?
# 2. Depot space binds. Imad offers to rent 10 cubic metres for SAR 500 for the week. Accept?
# 3. Why does the programme cut dry goods rather than dairy when space is short?
#
# <details><summary>Answers</summary>
#
# 1. Half a truck costs a whole trip; a continuous truck count would underprice small orders.
# 2. Compare SAR 500 with the saving the re-solve printed.
# 3. Cutting dry goods does two things at once: it frees depot space and lets supplier S1 fit one truck instead
#    of two (SAR 2,400). Dairy already fits its two reefers, so cutting it saves space only.
#
# </details>
#
# **Carry forward.** The plan is in `data/reference/d3_2.json`; Lab 3.4 stress-tests it.
