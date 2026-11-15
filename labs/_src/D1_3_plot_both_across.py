# %% [markdown]
# # Lab 1.3 · Learning curves with intervals
#
# Day 1 · Session 3 (Learning curves decide) · 25 min.
#
# One comparison at 2,000 orders is one point. Bustan has 500 orders for a new customer segment and 50,000
# for its oldest depots, so the question is where each approach wins. You score the tuned baseline and the
# pretrained model on repeated subsamples at five training sizes, put an interval on every point, add the
# supplied scores at 50,000 orders and read the curves.
#
# **Where to run it.** The pretrained model runs in Colab (see Lab 1.2); elsewhere the stand-in answers. The
# supplied scores at 50,000 orders were written by the course's run with the stand-in
# (`_tools/make_reference.py`): a 50,000-row in-context run needs a GPU.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/orders.csv.gz`, `data/reference/curve_supplied.json`.
#
# **Tasks**
# - [ ] 1 · The sizes and the repeats (4')
# - [ ] 2 · Score both models at each size (9')
# - [ ] 3 · Intervals per point (5')
# - [ ] 4 · Add the supplied largest size and plot (4')
# - [ ] 5 · Where each approach wins (3')
# - [ ] ✱ Stretch · Add the run time per size as a second panel

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-404.git"
    if not Path("/content/ds-404").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-404"], check=True)
    os.chdir("/content/ds-404/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt", "tabicl"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import time
import numpy as np
import pandas as pd
import bustan as U

plt = U.style()

# %% [markdown]
# ## 1 · The sizes and the repeats (4')
#
# Every point is a fresh subsample of the training pool, scored on the same held-out customers. Repeats with
# different seeds show how much a point moves when only the sample changes.

# %%
pool, test = U.curve_split()
Xt, yt = U.xy(test)
SIZES = U.pick([500, 1000, 2000, 5000, 10000], [500, 1000, 2000])
REPEATS = U.pick(3, 2)
BUDGET = U.pick(8, 4)                 # configurations per size for the baseline, three grouped folds each
_, info = U.tabular_model()
print("pretrained column answered by:", info["model"])
print(f"sizes {SIZES}, {REPEATS} repeats each, test {len(test):,} orders of held-out customers")

# %% [markdown]
# ## 2 · Score both models at each size (9')
#
# `U.tuned_ensemble(X, y, groups, Xtest, budget)` runs Lab 1.1's procedure in one call (random search under
# grouped folds, the top three averaged) and returns test probabilities.
#
# ✏️ **Exercise** · For one size `n` and repeat `r`: draw `n` orders from the pool with
# `random_state=U.SEED + r`, score the tuned ensemble and the pretrained model on the test set, and return a
# dict with `size`, `repeat`, `gbm_auc`, `pre_auc`, `gbm_seconds`, `pre_seconds`.

# %%
def score_point(n: int, r: int) -> dict:
    # BEGIN SOLUTION
    tr = pool.sample(n, random_state=U.SEED + r)
    X, y = U.xy(tr)
    t0 = time.perf_counter()
    p_gbm, _ = U.tuned_ensemble(X, y, tr.customer, Xt, budget=BUDGET, seed=U.SEED + r)
    t1 = time.perf_counter()
    m, _ = U.tabular_model()
    p_pre = m.fit(X, y).predict_proba(Xt)[:, 1]
    t2 = time.perf_counter()
    return {"size": n, "repeat": r, "gbm_auc": U.auc(yt, p_gbm), "pre_auc": U.auc(yt, p_pre),
            "gbm_seconds": t1 - t0, "pre_seconds": t2 - t1}
    # END SOLUTION

# %%
def ref_score_point(n, r):
    tr = pool.sample(n, random_state=U.SEED + r)
    X, y = U.xy(tr)
    t0 = time.perf_counter()
    p_gbm, _ = U.tuned_ensemble(X, y, tr.customer, Xt, budget=BUDGET, seed=U.SEED + r)
    t1 = time.perf_counter()
    p_pre = U.tabular_model()[0].fit(X, y).predict_proba(Xt)[:, 1]
    return {"size": n, "repeat": r, "gbm_auc": U.auc(yt, p_gbm), "pre_auc": U.auc(yt, p_pre),
            "gbm_seconds": t1 - t0, "pre_seconds": time.perf_counter() - t1}


try:
    score_point(SIZES[0], 0)
except NotImplementedError:
    score_point = ref_score_point
runs = pd.DataFrame([score_point(n, r) for n in SIZES for r in range(REPEATS)])
U.check(len(runs) == len(SIZES) * REPEATS, "one row per size and repeat")
print(runs.round(4).to_string(index=False))

# %% [markdown]
# ## 3 · Intervals per point (5')
#
# ✏️ **Exercise** · Turn the runs into one row per size and model with `mean`, `lo`, `hi` (a 95% t interval
# over the repeats, `U.mean_ci`). Also return the paired difference per size (pretrained minus baseline,
# repeat by repeat) with its interval, in columns `diff`, `diff_lo`, `diff_hi`.

# %%
def summarise(runs: pd.DataFrame):
    # BEGIN SOLUTION
    rows, diffs = [], []
    for n, g in runs.groupby("size"):
        for model, col in (("tuned LightGBM", "gbm_auc"), ("pretrained", "pre_auc")):
            m, lo, hi = U.mean_ci(g[col])
            rows.append({"size": n, "model": model, "mean": m, "lo": lo, "hi": hi})
        d, dlo, dhi = U.mean_ci(g.pre_auc - g.gbm_auc)
        diffs.append({"size": n, "diff": d, "diff_lo": dlo, "diff_hi": dhi})
    return pd.DataFrame(rows), pd.DataFrame(diffs)
    # END SOLUTION

# %%
def ref_summarise(runs):
    rows, diffs = [], []
    for n, g in runs.groupby("size"):
        for model, col in (("tuned LightGBM", "gbm_auc"), ("pretrained", "pre_auc")):
            m, lo, hi = U.mean_ci(g[col]); rows.append({"size": n, "model": model, "mean": m, "lo": lo, "hi": hi})
        d, dlo, dhi = U.mean_ci(g.pre_auc - g.gbm_auc); diffs.append({"size": n, "diff": d, "diff_lo": dlo, "diff_hi": dhi})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


try:
    curve, diffs = summarise(runs)
except NotImplementedError:
    summarise = ref_summarise
    curve, diffs = summarise(runs)
U.check(len(curve) == 2 * len(SIZES) and (curve.lo <= curve["mean"]).all(), "two points per size, each inside its interval")

# %% [markdown]
# ## 4 · Add the supplied largest size and plot (4')
#
# The file holds five repeats at 50,000 orders for both models. Add them with the same `summarise`, then plot
# AUC against training size on a log axis with the intervals as error bars.

# %%
sup = U.reference("curve_supplied")
print("supplied 50,000-order scores; pretrained column answered by:", sup["repeats"][0]["pretrained"])
big = pd.DataFrame(sup["repeats"]).assign(size=sup["size"])
all_runs = pd.concat([runs, big[runs.columns]], ignore_index=True)
curve, diffs = summarise(all_runs)
fig, ax = plt.subplots()
for model, c in (("tuned LightGBM", "tab:blue"), ("pretrained", "tab:orange")):
    d = curve[curve.model == model]
    ax.errorbar(d["size"], d["mean"], yerr=[d["mean"] - d.lo, d.hi - d["mean"]], marker="o", capsize=3, color=c,
                label=model if model != "pretrained" else f"pretrained ({'live' if info['live'] else 'stand-in'})")
ax.set_xscale("log"); ax.set_xlabel("training orders"); ax.set_ylabel("AUC on held-out customers")
ax.set_title("Late-delivery risk: learning curves with 95% intervals"); ax.legend()
plt.show()

# %% [markdown]
# ## 5 · Where each approach wins (3')
#
# A size is a win only when the interval of the paired difference stays on one side of zero.

# %%
diffs["verdict"] = np.where(diffs.diff_lo > 0, "pretrained wins", np.where(diffs.diff_hi < 0, "baseline wins", "cannot tell"))
print(diffs.round(4).to_string(index=False))
U.check(set(diffs.verdict) <= {"pretrained wins", "baseline wins", "cannot tell"}, "a verdict for every size")
U.save_reference("d1_3", {"answered_by": info["model"], "curve": curve.to_dict("records"), "diffs": diffs.to_dict("records"),
                          "runs": all_runs.to_dict("records")})

# %% [markdown]
# **Look at.** With three repeats the t interval is wide: "cannot tell" is an honest result, not a failure.
# More repeats narrow it; so would scoring on more test customers.

# %% [markdown]
# ## ✱ Stretch · Run time per size
#
# Add a second panel with `gbm_seconds` (tuning included) and `pre_seconds` against size. Where would the
# lines cross on a CPU, and how would a GPU runtime move the pretrained line?

# %% [markdown]
# ## Checkpoint
#
# 1. At 500 orders the pretrained model's mean is higher but the interval of the difference crosses zero.
#    What do you write in the memo?
# 2. Why are the repeats drawn as new subsamples instead of new folds of one subsample?
# 3. Why are the 50,000-order scores supplied rather than computed in the lab?
#
# <details><summary>Answers</summary>
#
# 1. That the two cannot be told apart at that size with this evidence, and how many repeats would decide it.
# 2. The question is how the method does on a sample of that size, so the sample itself must vary.
# 3. An in-context model at 50,000 rows needs a GPU and more than the lab's minutes; the file says which model
#    wrote it.
#
# </details>
#
# **Carry forward.** The curve and the verdicts are in `data/reference/d1_3.json`; the capstone reuses the method.
