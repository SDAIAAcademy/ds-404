# %% [markdown]
# # Lab 4.3 · Weighted conformal intervals
#
# Day 4 · Session 3 (Correcting for shift) · 25 min.
#
# Shadi's dispatchers want a range, not a guess: "this order will take between 9 and 30 hours, nine times in ten".
# You build that range for delivery hours with split conformal on the old depots, check how often it holds in the
# north, then weight the calibration orders by how northern they look and check again, by segment.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/orders.csv.gz`. The point model is a LightGBM regressor on the logarithm of delivery
# hours, trained on 70% of the old depots' customers; the other 30% calibrate.
#
# **Tasks**
# - [ ] 1 · Split conformal on old depots (6')
# - [ ] 2 · Coverage in the north (4')
# - [ ] 3 · Weights from the classifier (7')
# - [ ] 4 · Weighted intervals and their coverage (8')
# - [ ] ✱ Stretch · Clip the weights and watch width and coverage

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
pd.set_option("display.width", 160)
ALPHA = 0.10                                             # 90% intervals
o = U.orders()
old = o[o.region != "northern"].reset_index(drop=True)   # all old orders, quick mode too: fewer make the weights noisy
north = o[o.region == "northern"].reset_index(drop=True)
rng = np.random.default_rng(U.SEED)
cust = old.customer.unique()
cal_customers = set(rng.choice(cust, int(len(cust) * 0.3), replace=False))
is_cal = old.customer.isin(cal_customers).to_numpy()
Xo, _ = U.xy(old)
Xn, _ = U.xy(north)
yo, yn = np.log(old.delivery_hours.to_numpy()), np.log(north.delivery_hours.to_numpy())
model = U.lgbm_reg(n_jobs=1).fit(Xo[~is_cal], yo[~is_cal])
r_cal = np.abs(yo[is_cal] - model.predict(Xo[is_cal]))     # calibration scores: absolute residuals (log hours)
pred_n = model.predict(Xn)
r_n = np.abs(yn - pred_n)
print(f"train {(~is_cal).sum():,} | calibration {is_cal.sum():,} orders | north {len(north):,}")

# %% [markdown]
# ## 1 · Split conformal on old depots (6')
#
# Recap of DS-305: with n calibration scores, the interval half-width is the ⌈(n+1)(1-α)⌉-th smallest score.
# If calibration and new orders are exchangeable, the interval holds at least 90% of the time.
#
# ✏️ **Exercise** · Return the half-width `q` from the calibration scores.

# %%
def split_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    # BEGIN SOLUTION
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])
    # END SOLUTION

# %%
try:
    q = split_q(r_cal)
except NotImplementedError:
    q = float(np.sort(r_cal)[min(int(np.ceil((len(r_cal) + 1) * (1 - ALPHA))), len(r_cal)) - 1])
print(f"half-width {q:.3f} in log hours: the interval runs from {np.exp(-q):.2f}x to {np.exp(q):.2f}x the prediction")
U.check(abs((r_cal <= q).mean() - (1 - ALPHA)) < 0.01, "90% of calibration orders fall inside")

# %% [markdown]
# ## 2 · Coverage in the north (4')
#
# ✏️ **Exercise** · Return the share of northern orders whose delivery hours fall inside their interval, overall
# and by segment.

# %%
def coverage(r: np.ndarray, q, groups: pd.Series):
    # BEGIN SOLUTION
    inside = pd.Series(r <= q)
    return float(inside.mean()), inside.groupby(groups.to_numpy()).mean()
    # END SOLUTION

# %%
try:
    cov_split, seg_split = coverage(r_n, q, north.segment)
except NotImplementedError:
    coverage = lambda r, q, g: (float(np.mean(r <= q)), pd.Series(r <= q).groupby(g.to_numpy()).mean())
    cov_split, seg_split = coverage(r_n, q, north.segment)
print(f"north coverage, split conformal: {cov_split:.3f}")
print(seg_split.round(3).to_string())
U.check(cov_split < 1 - ALPHA, "the old intervals hold less than 90% of the time in the north")

# %% [markdown]
# ## 3 · Weights from the classifier (7')
#
# Weighted conformal (Tibshirani and others, 2019) reweights each calibration order by the density ratio
# w(x) = p(north | x) / p(old | x), times the ratio of sample sizes. A classifier gives the probability. Keep it
# simple (7 leaves, 100 trees) and score it out of fold, so that the weights are not overconfident.
#
# ✏️ **Exercise** · Return the weights of the calibration orders and of the northern orders.

# %%
def density_weights(X_cal, X_new):
    # BEGIN SOLUTION
    X = pd.concat([X_cal, X_new], ignore_index=True)
    y = np.r_[np.zeros(len(X_cal)), np.ones(len(X_new))]
    p = np.zeros(len(y))
    for tr, te in U.random_folds(len(y), 5):
        m = U.lgbm(n_estimators=100, num_leaves=7, n_jobs=1).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    p = np.clip(p, 1e-4, 1 - 1e-4)
    w = p / (1 - p) * len(X_cal) / len(X_new)
    return w[:len(X_cal)], w[len(X_cal):]
    # END SOLUTION

# %%
try:
    w_cal, w_new = density_weights(Xo[is_cal], Xn)
except NotImplementedError:
    X = pd.concat([Xo[is_cal], Xn], ignore_index=True); yy = np.r_[np.zeros(is_cal.sum()), np.ones(len(Xn))]; p = np.zeros(len(yy))
    for tr, te in U.random_folds(len(yy), 5):
        p[te] = U.lgbm(n_estimators=100, num_leaves=7, n_jobs=1).fit(X.iloc[tr], yy[tr]).predict_proba(X.iloc[te])[:, 1]
    p = np.clip(p, 1e-4, 1 - 1e-4); w = p / (1 - p) * is_cal.sum() / len(Xn)
    w_cal, w_new = w[:is_cal.sum()], w[is_cal.sum():]
ess = w_cal.sum() ** 2 / (w_cal ** 2).sum()
print(f"calibration weights: median {np.median(w_cal):.3f}, max {w_cal.max():.1f}")
print(f"effective sample size {ess:,.0f} of {len(w_cal):,} calibration orders")
U.check(len(w_cal) == is_cal.sum() and len(w_new) == len(north), "one weight per order")

# %% [markdown]
# **Look at.** A few hundred old orders that look northern (long distance, new drivers) carry most of the weight.
# The effective sample size says how many equally weighted orders they are worth.

# %% [markdown]
# ## 4 · Weighted intervals and their coverage (8')
#
# For a northern order with weight w_new, the half-width is the smallest calibration score at which the
# calibration weights below it reach 90% of (sum of calibration weights + w_new). The new order's own weight sits
# at infinity: if the calibration orders that look like it are too few, the interval is infinite, which is the
# honest answer.
#
# ✏️ **Exercise** · Return one half-width per northern order.

# %%
def weighted_q(r_cal, w_cal, w_new, alpha: float = ALPHA) -> np.ndarray:
    # BEGIN SOLUTION
    order = np.argsort(r_cal)
    r_sorted, cw = r_cal[order], np.cumsum(w_cal[order])
    k = np.searchsorted(cw, (1 - alpha) * (cw[-1] + w_new))
    return np.where(k < len(r_sorted), r_sorted[np.minimum(k, len(r_sorted) - 1)], np.inf)
    # END SOLUTION

# %%
try:
    q_w = weighted_q(r_cal, w_cal, w_new)
except NotImplementedError:
    order = np.argsort(r_cal); rs, cw = r_cal[order], np.cumsum(w_cal[order])
    k = np.searchsorted(cw, (1 - ALPHA) * (cw[-1] + w_new)); q_w = np.where(k < len(rs), rs[np.minimum(k, len(rs) - 1)], np.inf)
    coverage = lambda r, q, g: (float(np.mean(r <= q)), pd.Series(r <= q).groupby(g.to_numpy()).mean())
cov_w, seg_w = coverage(r_n, q_w, north.segment)
_, dep_split = coverage(r_n, q, north.depot)
_, dep_w = coverage(r_n, q_w, north.depot)
table = pd.DataFrame({"split": pd.concat([seg_split, dep_split]), "weighted": pd.concat([seg_w, dep_w])})
table.loc["all northern orders"] = [cov_split, cov_w]
print(table.round(3).to_string())
finite = np.isfinite(q_w)
width = lambda h: np.exp(pred_n + h) - np.exp(pred_n - h)
print(f"\ninfinite intervals: {np.mean(~finite):.1%}")
print(f"median width in hours: split {np.median(width(q)):.1f} | weighted {np.median(width(q_w[finite])):.1f}")
U.check(cov_w > cov_split, "weighting raises coverage in the north")
U.save_reference("d4_3", {"q_split": q, "cover_split": cov_split, "cover_weighted": cov_w, "ess": ess, "n_cal": int(is_cal.sum()),
                          "table": table.reset_index(names="group").to_dict("records"), "infinite_share": float(np.mean(~finite)),
                          "width_split": float(np.median(width(q))), "width_weighted": float(np.median(width(q_w[finite])))})

fig, ax = plt.subplots(figsize=(7, 3.2))
table.drop("all northern orders")[["split", "weighted"]].plot.barh(ax=ax)
ax.axvline(1 - ALPHA, color="k", lw=1, ls="--")
ax.set_xlim(0.6, 1)
ax.set_xlabel("coverage of 90% intervals in the north")
plt.show()

# %% [markdown]
# **Look at.** Weighting brings the overall coverage back to about 90%, with wider intervals. It cannot fix a
# segment whose link between features and hours has changed: institutional customers stay below target. Weights
# correct covariate shift; concept shift needs northern outcomes.

# %% [markdown]
# ## ✱ Stretch · Clip the weights
#
# Clip every weight at 10, then at 2, and recompute coverage, the median width and the effective sample size.
# What does clipping trade?

# %% [markdown]
# ## Checkpoint
#
# 1. What assumption does split conformal need, and which Lab 4.1 result breaks it?
# 2. Why are the classifier's probabilities scored out of fold?
# 3. Coverage is back near 90% overall. What do you still tell Shadi?
#
# <details><summary>Answers</summary>
#
# 1. Calibration and new orders exchangeable; the adversarial AUC of 0.95 says they are not.
# 2. In-sample probabilities are overconfident near 0 and 1; their ratios explode into a few huge weights.
# 3. That the guarantee is overall, not per segment: institutional orders still fall outside more often, until
#    northern outcomes recalibrate the intervals.
#
# </details>
