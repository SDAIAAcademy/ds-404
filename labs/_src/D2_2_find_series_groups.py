# %% [markdown]
# # Lab 2.2 · Where the benchmark wins
#
# Day 2 · Session 2 (Where zero-shot loses) · 25 min.
#
# One average over 400 series hides the groups where a method fails. You classify each series from its own
# history (short, intermittent, smooth, and the Ramadan items from the product master), score every method by
# group, find where a benchmark beats the zero-shot forecaster by more than the noise, and write a routing
# rule chosen on earlier origins only.
#
# **Where to run it.** This lab scores the supplied zero-shot forecasts (`data/reference/zero_shot.npz`),
# written by the course's run. In Colab you can replace them with your own Chronos-2 run from Lab 2.1.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/demand.npz`, `data/series_meta.csv`, `data/reference/benchmarks.npz`,
# `data/reference/zero_shot.npz`.
#
# **Tasks**
# - [ ] 1 · Classify the series (7')
# - [ ] 2 · Scores by group (7')
# - [ ] 3 · Where does the benchmark win? (6')
# - [ ] 4 · A routing rule (5')
# - [ ] ✱ Stretch · Score the routed forecast as a fifth method

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

plt = U.style()

S = U.series_subset()
Y = U.demand().iloc[:, S]
B, _ = U.benchmarks()
Z, who = U.zero_shot_supplied()
meta = U.meta().iloc[S].reset_index(drop=True)
print("zero-shot forecasts answered by:", who)
fc_all = {"seasonal naive": B[0][:, S], "smoothing": B[1][:, S], "LightGBM (tuned)": B[2][:, S], "zero-shot": Z[0][:, S]}

# %% [markdown]
# ## 1 · Classify the series (7')
#
# Two numbers from the last 365 days before the first origin describe a demand pattern: **ADI**,
# the average number of days between non-zero days, and **CV squared** of the non-zero sizes. A series with
# fewer than 180 days of history is **short**; in this lab, ADI of 1.32 or more is **intermittent** (CV squared
# is printed for reading, not used); the rest are **regular**. Ramadan items come from the product master (`category`).
#
# ✏️ **Exercise** · Return a table with one row per series: `series`, `days`, `adi`, `cv2`, `cls`.

# %%
def classify(Y: pd.DataFrame, until) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for c in Y.columns:
        h = Y[c].loc[:until].dropna()
        last = h.iloc[-365:].to_numpy()
        nz = last[last > 0]
        adi = len(last) / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
        cls = "short" if len(h) < 180 else ("intermittent" if adi >= 1.32 else "regular")
        rows.append({"series": c, "days": len(h), "adi": adi, "cv2": cv2, "cls": cls})
    return pd.DataFrame(rows)
    # END SOLUTION

# %%
FIRST = U.ORIGINS[0]
try:
    cls = classify(Y, FIRST)
except NotImplementedError:
    rows = []
    for c in Y.columns:
        h = Y[c].loc[:FIRST].dropna(); last = h.iloc[-365:].to_numpy(); nz = last[last > 0]
        adi = len(last) / max(len(nz), 1)
        rows.append({"series": c, "days": len(h), "adi": adi, "cv2": (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan,
                     "cls": "short" if len(h) < 180 else ("intermittent" if adi >= 1.32 else "regular")})
    cls = pd.DataFrame(rows)
cls["ramadan_item"] = meta.category.str.endswith("ramadan").to_numpy()
cls["group"] = np.where(cls.cls != "regular", cls.cls, np.where(cls.ramadan_item, "Ramadan items", "regular"))
print(cls.group.value_counts().to_string())
print(pd.crosstab(cls.group, meta.group.to_numpy()).to_string())
U.check(set(cls.group) == {"short", "intermittent", "Ramadan items", "regular"}, "four groups found")

# %% [markdown]
# **Look at.** The crosstab compares your classes with the generator's answer key (`series_meta.group`).
# Products launched less than 180 days before the first origin are short there, even when they will be
# regular by the last origin: a class depends on the date you classify at.

# %% [markdown]
# ## 2 · Scores by group (7')
#
# `U.score_long` scores every model at every origin and series. Join your groups and average.

# %%
scored_i = [list(U.ORIGINS).index(o) for o in U.SCORED]
long = U.score_long(Y, {k: v[scored_i] for k, v in fc_all.items()}, U.SCORED)
long = long.merge(cls[["series", "group"]], on="series", suffixes=("_key", ""))
sql_by_group = long.pivot_table(index="group", columns="model", values="sql", aggfunc="mean")
mase_by_group = long.pivot_table(index="group", columns="model", values="mase", aggfunc="mean")
print("scaled quantile loss by group\n", sql_by_group.round(3).to_string())
print("\nscaled point error by group\n", mase_by_group.round(3).to_string())
U.check(sql_by_group.shape == (4, 4), "four groups by four models")

# %% [markdown]
# ## 3 · Where does the benchmark win? (6')
#
# ✏️ **Exercise** · For each group and each benchmark, the paired difference in `sql` (zero-shot minus
# benchmark, per series and origin) with a bootstrap interval. Return a table with `group`, `benchmark`,
# `diff`, `lo`, `hi`, `benchmark_wins` (True when `lo` > 0).

# %%
def wins_table(long: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    w = long.pivot_table(index=["group", "series", "origin"], columns="model", values="sql").reset_index()
    rows = []
    for g, d in w.groupby("group"):
        for b in ("seasonal naive", "smoothing", "LightGBM (tuned)"):
            m, lo, hi = U.bootstrap_ci(d["zero-shot"] - d[b])
            rows.append({"group": g, "benchmark": b, "diff": m, "lo": lo, "hi": hi, "benchmark_wins": lo > 0})
    return pd.DataFrame(rows)
    # END SOLUTION

# %%
try:
    wins = wins_table(long)
except NotImplementedError:
    w = long.pivot_table(index=["group", "series", "origin"], columns="model", values="sql").reset_index()
    wins = pd.DataFrame([{"group": g, "benchmark": b, **dict(zip(["diff", "lo", "hi"], U.bootstrap_ci(d["zero-shot"] - d[b])))}
                         for g, d in w.groupby("group") for b in ("seasonal naive", "smoothing", "LightGBM (tuned)")])
    wins["benchmark_wins"] = wins.lo > 0
print(wins.round(3).to_string(index=False))
lost = wins[wins.benchmark_wins & (wins.benchmark != "LightGBM (tuned)")]
print("\ngroups where a statistical benchmark beats the zero-shot forecaster:", sorted(set(lost.group)) or "none")
gap = wins[wins.benchmark == "LightGBM (tuned)"].set_index("group")["diff"].sort_values()
print("zero-shot's gap to the tuned LightGBM, smallest first:", gap.round(3).to_dict())
U.check(wins.benchmark_wins.any(), "at least one group where a benchmark wins beyond the noise")

# %% [markdown]
# ## 4 · A routing rule (5')
#
# A rule chosen on the same windows it is scored on flatters itself. Choose the best method per group on the
# two **earlier** origins (2025-10-05 and 2025-11-16) and only then look at the scored ones.
#
# ✏️ **Exercise** · Return a dict `{group: model}` with the lowest mean `sql` per group on the earlier origins.
# A group with no series scored at the earlier origins (products not launched yet) gets `default`: a method
# that needs no history of its own.

# %%
early_long = U.score_long(Y, {k: v[:2] for k, v in fc_all.items()}, U.ORIGINS[:2]).merge(
    cls[["series", "group"]], on="series", suffixes=("_key", ""))


def routing_rule(early_long: pd.DataFrame, groups, default: str = "zero-shot") -> dict:
    # BEGIN SOLUTION
    rule = early_long.groupby(["group", "model"]).sql.mean().unstack().idxmin(axis=1).to_dict()
    return {g: rule.get(g, default) for g in groups}
    # END SOLUTION

# %%
try:
    rule = routing_rule(early_long, sorted(set(cls.group)))
except NotImplementedError:
    r0 = early_long.groupby(["group", "model"]).sql.mean().unstack().idxmin(axis=1).to_dict()
    rule = {g: r0.get(g, "zero-shot") for g in sorted(set(cls.group))}
print("routing rule (chosen on the two earlier origins):")
for g, m in rule.items():
    print(f"  {g:>14} -> {m}   (scored origins: {sql_by_group.loc[g, m]:.3f}; zero-shot {sql_by_group.loc[g, 'zero-shot']:.3f})")
U.check(set(rule) == set(cls.group), "a model for every group")
U.save_reference("d2_2", {"answered_by": who, "counts": cls.group.value_counts().to_dict(),
                          "sql_by_group": sql_by_group.reset_index().to_dict("records"),
                          "mase_by_group": mase_by_group.reset_index().to_dict("records"),
                          "wins": wins.to_dict("records"), "rule": rule})

# %% [markdown]
# ## ✱ Stretch · The routed forecast
#
# Build the routed forecast (each series takes its group's model) at the scored origins and score it with
# `U.score_long`. Does it beat the best single model, and by more than its interval?

# %% [markdown]
# ## Checkpoint
#
# 1. Over all series the zero-shot forecaster is second best. In the short group it ties smoothing and trails
#    the tuned LightGBM most. What do you tell the planner who manages new products?
# 2. Why is the routing rule chosen on earlier origins?
# 3. A group holds 12 series. Its difference is large but the interval crosses zero. Act on it?
#
# <details><summary>Answers</summary>
#
# 1. For new products the tuned LightGBM is the method to beat; the zero-shot model is no better than smoothing
#    there. Test the real pretrained model in Colab before changing anything.
# 2. A rule picked on the scored windows is tested on its own training data.
# 3. Not yet: gather more origins first, or treat it as a hypothesis for the capstone.
#
# </details>
#
# **Carry forward.** The group table and the rule are in `data/reference/d2_2.json`.
