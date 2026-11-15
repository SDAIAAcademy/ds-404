# %% [markdown]
# # Lab 3.3 · Regret against error
#
# Day 3 · Session 3 (Scoring by regret) · 25 min.
#
# Ahlam Al-Kuwaileet must choose one forecaster for the weekly orders of the Riyadh depot. Day 2 ranked them
# by error. Here you rank them by what their orders cost: for twelve weeks (three origins, four weeks each),
# each forecaster's quantiles become newsvendor orders for the 20 products, the orders meet the demand that
# happened, and you compare the ranking by regret with the ranking by error.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** The supplied forecasts of three forecasters (`data/reference/benchmarks.npz`: smoothing
# and the tuned LightGBM; `data/reference/zero_shot.npz`: the zero-shot forecaster with covariates, written by
# the stand-in) at the origins 2026-01-04, 2026-03-15 and 2026-05-10, and the costs of Lab 3.1
# (`U.day3_products()`).
#
# **Tasks**
# - [ ] 1 · Plans from three forecasters (6')
# - [ ] 2 · Realised cost (5')
# - [ ] 3 · Regret against hindsight (6')
# - [ ] 4 · Two rankings, side by side (8')
# - [ ] ✱ Stretch · Repeat with a lower shortage cost: does the ranking change?

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
import bustan as U

pd.set_option("display.width", 160)
Y = U.demand()
P = U.day3_products()
COLS = list(Y.columns)
IDX = [COLS.index(s) for s in P.series]
FORECASTERS = ["smoothing", "lightgbm", "zero_shot_cov"]
ORIGIN_IDX = [2, 4, 5]                                  # 2026-01-04, 2026-03-15 (Eid al-Fitr inside), 2026-05-10
_, answered = U.zero_shot_supplied()
print("zero_shot_cov answered by:", answered)

# %% [markdown]
# ## 1 · Plans from three forecasters (6')
#
# `U.week_inputs(model, origin_index, week)` turns one forecaster's daily quantiles into the week's newsvendor
# order (column `target`), as you did by hand in Lab 3.1. Here the orders are not cut by capacity, so that the
# comparison is about the forecasts alone.
#
# ✏️ **Exercise** · Return one row per forecaster, origin, week (0 to 3) and product with the columns
# `model`, `origin`, `week`, `product`, `category`, `underage`, `overage` and `order`.

# %%
def plans(forecasters, origin_idx) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for m in forecasters:
        for oi in origin_idx:
            for w in range(4):
                wi = U.week_inputs(m, oi, w)
                rows.append(pd.DataFrame({"model": m, "origin": U.ORIGINS[oi].date(), "week": w,
                                          "product": wi["product"], "category": wi.category,
                                          "underage": wi.underage, "overage": wi.overage, "order": wi.target}))
    return pd.concat(rows, ignore_index=True)
    # END SOLUTION

# %%
def ref_plans(forecasters, origin_idx):
    rows = []
    for m in forecasters:
        for oi in origin_idx:
            for w in range(4):
                wi = U.week_inputs(m, oi, w)
                rows.append(pd.DataFrame({"model": m, "origin": U.ORIGINS[oi].date(), "week": w, "product": wi["product"],
                                          "category": wi.category, "underage": wi.underage, "overage": wi.overage,
                                          "order": wi.target}))
    return pd.concat(rows, ignore_index=True)


try:
    plan = plans(FORECASTERS, ORIGIN_IDX)
except NotImplementedError:
    plan = ref_plans(FORECASTERS, ORIGIN_IDX)
U.check(len(plan) == 3 * 3 * 4 * 20, "720 orders: 3 forecasters x 12 weeks x 20 products")
print(plan.pivot_table(index="product", columns="model", values="order", aggfunc="mean").head(8).round(0).to_string())

# %% [markdown]
# ## 2 · Realised cost (5')
#
# The demand that happened in each week is the sum of seven days of `U.actuals`.
#
# ✏️ **Exercise** · Add the column `demand` (cases in that week) and `cost`: underage times the shortfall plus
# overage times the leftover.

# %%
def weekly_demand() -> pd.DataFrame:
    rows = []
    for oi in ORIGIN_IDX:
        y = U.actuals(Y, U.ORIGINS[oi])[IDX]
        for w in range(4):
            rows.append(pd.DataFrame({"origin": U.ORIGINS[oi].date(), "week": w, "product": P["product"],
                                      "demand": np.nansum(y[:, 7 * w:7 * w + 7], axis=1)}))
    return pd.concat(rows, ignore_index=True)


def realised_cost(plan: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    t = plan.merge(demand, on=["origin", "week", "product"])
    t["cost"] = t.underage * np.maximum(t.demand - t.order, 0) + t.overage * np.maximum(t.order - t.demand, 0)
    return t
    # END SOLUTION

# %%
D = weekly_demand()
try:
    plan = realised_cost(plan, D)
except NotImplementedError:
    plan = plan.merge(D, on=["origin", "week", "product"])
    plan["cost"] = plan.underage * np.maximum(plan.demand - plan.order, 0) + plan.overage * np.maximum(plan.order - plan.demand, 0)
U.check({"demand", "cost"} <= set(plan.columns) and (plan.cost >= 0).all(), "every order has a realised cost")
print(plan.groupby("model").cost.sum().round(0).to_string())

# %% [markdown]
# ## 3 · Regret against hindsight (6')
#
# **Regret** is the realised cost minus the cost of the best decision made knowing the demand. With no capacity
# limit and demand in whole cases, the hindsight order is the demand itself and costs nothing, so here regret
# equals realised cost. (With the depot limit of Lab 3.2 the hindsight plan would itself cost something.)
# A second, sharper view: regret against the best of the three forecasters **in that week**.
#
# ✏️ **Exercise** · Return one row per forecaster and week (`model`, `origin`, `week`) with `regret` (sum over
# products) and `regret_vs_best` (that week's regret minus the lowest regret of the three that week).

# %%
def weekly_regret(plan: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    hindsight = 0.0                                   # order = demand: no shortfall, no leftover
    wk = plan.groupby(["model", "origin", "week"], as_index=False).cost.sum()
    wk["regret"] = wk.cost - hindsight
    wk["regret_vs_best"] = wk.regret - wk.groupby(["origin", "week"]).regret.transform("min")
    return wk.drop(columns="cost")
    # END SOLUTION

# %%
try:
    wk = weekly_regret(plan)
except NotImplementedError:
    wk = plan.groupby(["model", "origin", "week"], as_index=False).cost.sum().rename(columns={"cost": "regret"})
    wk["regret_vs_best"] = wk.regret - wk.groupby(["origin", "week"]).regret.transform("min")
U.check(len(wk) == 36 and (wk.regret_vs_best >= 0).all(), "36 forecaster-weeks with their regret")
print(wk.pivot_table(index=["origin", "week"], columns="model", values="regret").round(0).to_string())
print("\nregret by product category, SAR:")
print(plan.pivot_table(index="category", columns="model", values="cost", aggfunc="sum").round(0).to_string())

# %% [markdown]
# **Look at.** Which category carries most of the regret? A chilled leftover loses 60% of its cost, so a
# forecaster that is too high on dairy pays for it, whatever its accuracy on dry goods.

# %% [markdown]
# ## 4 · Two rankings, side by side (8')
#
# The error of each forecaster on the same weeks and products: the scaled quantile loss (`sql`, Day 2's
# measure) of its daily forecasts on the seven days of each week.
#
# ✏️ **Exercise** · Return, per forecaster and week, `sql` (mean over the 20 products). Then compare the
# rankings: overall, per origin, and per week.

# %%
def weekly_error(forecasters, origin_idx) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for m in forecasters:
        for oi in origin_idx:
            o = U.ORIGINS[oi]
            y = U.actuals(Y, o)[IDX]
            q = U.daily_forecasts(m, oi)
            sc = np.array([U.scale_of(Y[c].loc[:o].to_numpy()) for c in P.series])
            for w in range(4):
                sl = slice(7 * w, 7 * w + 7)
                rows.append({"model": m, "origin": o.date(), "week": w, "sql": float(np.nanmean(U.scores(y[:, sl], q[:, :, sl], sc)["sql"]))})
    return pd.DataFrame(rows)
    # END SOLUTION

# %%
def ref_weekly_error(forecasters, origin_idx):
    rows = []
    for m in forecasters:
        for oi in origin_idx:
            o = U.ORIGINS[oi]; y = U.actuals(Y, o)[IDX]; q = U.daily_forecasts(m, oi)
            sc = np.array([U.scale_of(Y[c].loc[:o].to_numpy()) for c in P.series])
            for w in range(4):
                sl = slice(7 * w, 7 * w + 7)
                rows.append({"model": m, "origin": o.date(), "week": w, "sql": float(np.nanmean(U.scores(y[:, sl], q[:, :, sl], sc)["sql"]))})
    return pd.DataFrame(rows)


try:
    err = weekly_error(FORECASTERS, ORIGIN_IDX)
except NotImplementedError:
    err = ref_weekly_error(FORECASTERS, ORIGIN_IDX)
both = wk.merge(err, on=["model", "origin", "week"])

overall = both.groupby("model").agg(regret=("regret", "sum"), sql=("sql", "mean"))
overall["rank_regret"], overall["rank_error"] = overall.regret.rank().astype(int), overall.sql.rank().astype(int)
print("twelve weeks:"); print(overall.round(3).to_string())
by_origin = both.groupby(["origin", "model"]).agg(regret=("regret", "sum"), sql=("sql", "mean")).round(3)
print("\nper origin:"); print(by_origin.to_string())

both["r_regret"] = both.groupby(["origin", "week"]).regret.rank()
both["r_error"] = both.groupby(["origin", "week"]).sql.rank()
winners = both[both.r_regret == 1][["origin", "week", "model"]].merge(
    both[both.r_error == 1][["origin", "week", "model"]], on=["origin", "week"], suffixes=("_by_regret", "_by_error")
).sort_values(["origin", "week"])
differ = int((winners.model_by_regret != winners.model_by_error).sum())
print("\nweekly winners:"); print(winners.to_string(index=False))
print(f"\nthe week's best forecaster differs between the two rankings in {differ} of 12 weeks")

# paired: is the overall winner's lower regret more than week-to-week noise?
a = both.pivot_table(index=["origin", "week"], columns="model", values="regret")
first, second = overall.sort_values("regret").index[:2]
d = (a[first] - a[second]).to_numpy()
mean, lo, hi = U.bootstrap_ci(d)
print(f"{first} minus {second}, regret per week: SAR {mean:,.0f} (95% interval {lo:,.0f} to {hi:,.0f}) over 12 weeks")
U.check(differ >= 1, "the two rankings do not always agree")
U.save_reference("d3_3", {"overall": overall.reset_index().to_dict("records"), "by_origin": by_origin.reset_index().astype({"origin": str}).to_dict("records"),
                          "weekly_winners": winners.astype({"origin": str}).to_dict("records"), "differ": differ,
                          "paired": {"first": first, "second": second, "mean": mean, "lo": lo, "hi": hi},
                          "by_category": plan.pivot_table(index="category", columns="model", values="cost", aggfunc="sum").round(0).to_dict()})

# %% [markdown]
# **Look at.** Over twelve weeks the two rankings agree on the order. Week by week they do not, and at the
# origin 2026-05-10 the forecaster with the lower error has the higher regret. Error weighs the 0.1, 0.5 and
# 0.9 quantiles equally on every product; the orders use the 0.9 quantile of dry goods and a low quantile of
# dairy, and a SAR of dairy leftover weighs far more than a SAR of rice error.

# %% [markdown]
# ## ✱ Stretch · A lower shortage cost
#
# Drop the goodwill part: underage = the margin only (two thirds of the current value). Recompute the critical
# ratios and the orders from `mu` and `sd` of `U.week_inputs`, and rank again. Which forecaster gains?

# %% [markdown]
# ## Checkpoint
#
# 1. Why is regret equal to realised cost in this lab, and when would it not be?
# 2. A forecaster has lower error but higher regret at one origin. Name one reason.
# 3. Ahlam asks which forecaster to adopt. What do you report besides the overall ranking?
#
# <details><summary>Answers</summary>
#
# 1. With no capacity limit, the hindsight order equals the demand and costs nothing. Under the depot limit,
#    the best plan in hindsight still leaves some demand unmet, so regret is smaller than cost.
# 2. Its errors fall where they cost most: too high on chilled dairy, where a leftover loses 60% of its cost,
#    or too low on dry goods ordered at the 0.9 quantile.
# 3. The paired difference per week with its interval, and the regret by category: a decision should not
#    rest on a gap the weeks cannot separate from noise.
#
# </details>
