# %% [markdown]
# # Lab 2.1 · Zero-shot against the supplied backtests
#
# Day 2 · Session 1 (Zero-shot against a benchmark floor) · 25 min.
#
# Bustan's planners already have rolling-origin backtests of three methods on 400 daily demand series
# (100 products at four depots): a weekly seasonal naive (the floor), a smoothing model and a tuned global
# LightGBM. You add a zero-shot pretrained forecaster on the same four origins and 28-day horizon and score
# all four the same way: scaled point error, scaled quantile loss and the coverage of the 80% band.
#
# **Where to run it.** In Google Colab the setup cell installs `chronos-forecasting` and Chronos-2's weights
# download into the Colab runtime. Nothing downloads to your laptop. Elsewhere a **stand-in** answers: each
# series' weekday profile over its last eight weeks, with quantiles from its own recent errors. It is not a
# pretrained model, and every result says which one answered.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/demand.npz`, `data/calendar.csv`, `data/reference/benchmarks.npz` (the supplied
# backtests). Nothing from Day 1.
#
# **Tasks**
# - [ ] 1 · Who answers, and the supplied backtests (3')
# - [ ] 2 · Zero-shot at four origins (7')
# - [ ] 3 · Scaled error and quantile loss (8')
# - [ ] 4 · Against the floor (4')
# - [ ] 5 · Coverage of the 80% band (3')
# - [ ] ✱ Stretch · Shorten the context to 120 days and score again

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-404.git"
    if not Path("/content/ds-404").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-404"], check=True)
    os.chdir("/content/ds-404/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt", "chronos-forecasting"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import bustan as U

plt = U.style()

# %% [markdown]
# ## 1 · Who answers, and the supplied backtests (3')
#
# `U.benchmarks()` returns an array `[model, origin, series, quantile, day]`: three models, six origins,
# 400 series, the quantiles 0.1, 0.5 and 0.9, 28 days. Today you score the last four origins.

# %%
S = U.series_subset()
Y = U.demand().iloc[:, S]
B, names = U.benchmarks()
B = B[:, :, S]
scored = [list(U.ORIGINS).index(o) for o in U.SCORED]
print(f"{Y.shape[1]} series, {len(Y):,} days from {Y.index[0].date()} to {Y.index[-1].date()}")
print("benchmarks:", U.MODELS, "| scored origins:", [str(o.date()) for o in U.SCORED])
U.check(B.shape[2] == Y.shape[1] and names[S[0]] == Y.columns[0], "the backtests line up with the series")

# %% [markdown]
# ## 2 · Zero-shot at four origins (7')
#
# The model's only input is each series' history up to the origin (`U.contexts_at`): no product, no depot,
# no training on Bustan's data.
#
# ✏️ **Exercise** · Forecast every series at every scored origin with `U.zero_shot(contexts, index)`. Return
# an array `[origin, series, 3, 28]` and the `info` of the last call.

# %%
def zero_shot_backtest(Y: pd.DataFrame, origins) -> tuple:
    # BEGIN SOLUTION
    out, info = [], None
    for o in origins:
        ctx, idx = U.contexts_at(Y, o)
        q, info = U.zero_shot(ctx, idx)
        out.append(q)
    return np.stack(out), info
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    ZS, info = zero_shot_backtest(Y, U.SCORED)
except NotImplementedError:
    Zsup, who = U.zero_shot_supplied()
    ZS, info = Zsup[0][scored][:, S], {"model": who + " (supplied file)", "live": False, "seconds": 0.0}
print("answered by:", info["model"])
if not info["live"]:
    print("!! STAND-IN: these are not a pretrained model's forecasts. Run this notebook in Colab for Chronos-2's.")
U.check(ZS.shape == (len(U.SCORED), Y.shape[1], 3, U.H), "one forecast per origin and series, three quantiles, 28 days")
U.check(np.all(ZS[:, :, 0] <= ZS[:, :, 1] + 1e-9) and np.all(ZS[:, :, 1] <= ZS[:, :, 2] + 1e-9), "quantiles do not cross")

# %% [markdown]
# ## 3 · Scaled error and quantile loss (8')
#
# Each series is scored in its own units, divided by its **scale**: the mean absolute error of the weekly
# naive over its last year (`U.scale_of`). Then 1.0 means "as bad as last week's copy, in-sample".
#
# ✏️ **Exercise** · For one origin, score forecasts `q [series, 3, 28]` against the actual 28 days: return
# per-series `mase` (median against actual), `sql` (twice the mean pinball loss over the three quantiles,
# divided by the scale) and `cover80` (share of days inside the 0.1 to 0.9 band). `U.pinball(y, q, tau)` is given.

# %%
def score_origin(Y: pd.DataFrame, origin, q: np.ndarray) -> pd.DataFrame:
    # BEGIN SOLUTION
    y = U.actuals(Y, origin)
    sc = np.array([U.scale_of(Y[c].loc[:origin].to_numpy()) for c in Y.columns])
    mase = np.nanmean(np.abs(y - q[:, 1]), axis=1) / sc
    sql = 2 * np.nanmean([U.pinball(y, q[:, j], t) for j, t in enumerate(U.QL)], axis=(0, 2)) / sc
    cover = np.nanmean(np.where(np.isnan(y), np.nan, (y >= q[:, 0]) & (y <= q[:, 2])), axis=1)
    return pd.DataFrame({"series": Y.columns, "mase": mase, "sql": sql, "cover80": cover})
    # END SOLUTION

# %%
def ref_score_origin(Y, origin, q):
    s = U.scores(U.actuals(Y, origin), q, np.array([U.scale_of(Y[c].loc[:origin].to_numpy()) for c in Y.columns]))
    return pd.DataFrame({"series": Y.columns, **s})


fc = {"seasonal naive": B[0][scored], "smoothing": B[1][scored], "LightGBM (tuned)": B[2][scored], "zero-shot": ZS}
try:
    score_origin(Y, U.SCORED[0], ZS[0])
except NotImplementedError:
    score_origin = ref_score_origin
long = pd.concat([score_origin(Y, o, q[i]).assign(model=m, origin=o) for m, q in fc.items() for i, o in enumerate(U.SCORED)])
long = long.dropna(subset=["mase"])
check = U.score_long(Y, {"zero-shot": ZS}, U.SCORED)
U.check(np.allclose(long[long.model == "zero-shot"].sql.mean(), check.sql.mean(), rtol=1e-6), "your scores match the helper's")
table = long.groupby("model")[["mase", "sql", "cover80"]].mean()
print(table.round(3).to_string())
print(f"series-origins scored: {len(long) // len(fc):,} (products not yet launched at an origin are left out)")

# %% [markdown]
# ## 4 · Against the floor (4')
#
# ✏️ **Exercise** · Divide each model's mean `mase` and `sql` by the seasonal naive's. Below 1 beats the floor.

# %%
def relative_to_floor(table: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    return (table[["mase", "sql"]] / table.loc["seasonal naive", ["mase", "sql"]]).rename(columns=lambda c: c + "_vs_floor")
    # END SOLUTION

# %%
try:
    rel = relative_to_floor(table)
except NotImplementedError:
    rel = (table[["mase", "sql"]] / table.loc["seasonal naive", ["mase", "sql"]]).rename(columns=lambda c: c + "_vs_floor")
print(rel.round(3).to_string())
U.check(rel.loc["zero-shot", "sql_vs_floor"] < 1, "the zero-shot forecaster beats the floor on quantile loss")
best = rel.sql_vs_floor.idxmin()
print("lowest quantile loss:", best)

# %% [markdown]
# ## 5 · Coverage of the 80% band (3')
#
# An 80% band should hold the actual on about 80% of days. Read the coverage column: which methods'
# bands are too narrow, and what would a planner who trusts them run short of?

# %%
cov = table.cover80.sort_values()
print(cov.round(3).to_string())
U.check(cov.between(0, 1).all(), "coverage is a share")
U.save_reference("d2_1", {"answered_by": info["model"], "table": table.reset_index().to_dict("records"),
                          "rel": rel.reset_index().to_dict("records")})

# %% [markdown]
# ## ✱ Stretch · A shorter context
#
# Run the zero-shot model with `U.contexts_at(Y, o, length=120)` and score again. Which group of series loses
# most when last year's Ramadan falls out of the context?

# %% [markdown]
# ## Checkpoint
#
# 1. The zero-shot forecaster beats the floor but not the tuned LightGBM. Is it useless?
# 2. Why is every series divided by its own scale before averaging?
# 3. A method has the lowest quantile loss and 65% coverage of its 80% band. What do you report?
#
# <details><summary>Answers</summary>
#
# 1. No: it needed no training, no features and no tuning. Its value depends on where it wins (Lab 2.2) and on
#    what it adds to an ensemble (Lab 2.4).
# 2. Otherwise the fast-moving staples, in thousands of cases, decide the average alone.
# 3. Both numbers: a low loss with bands that are too narrow; Day 3 shows what narrow bands cost.
#
# </details>
#
# **Carry forward.** The score table is in `data/reference/d2_1.json`.
