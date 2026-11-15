# %% [markdown]
# # Lab 4.1 · Adversarial validation
#
# Day 4 · Session 1 (Detecting shift) · 25 min.
#
# Bustan opened the northern region (Hail and Tabuk depots) on 24 August 2025. Shadi Al-Baiz, who runs it, wants
# the late-delivery model of the old depots to warn his dispatchers too. Before anyone trusts it there, you test
# whether the northern orders look like the orders the model learned from: a classifier that tries to tell
# them apart. If it can, the data has shifted, and its most useful features say where.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/orders.csv.gz` (80,000 orders: 72,000 in the old regions since July 2023, 8,000 in
# the north since August 2025).
#
# **Tasks**
# - [ ] 1 · Two samples: old depots and the north (3')
# - [ ] 2 · The adversarial classifier (8')
# - [ ] 3 · Which features carry the shift (6')
# - [ ] 4 · Label and concept shift, checked apart (8')
# - [ ] ✱ Stretch · Drop the top shifted feature and run again

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
o = U.orders()
NORTH_START = pd.Timestamp("2025-08-24")

# %% [markdown]
# ## 1 · Two samples: old depots and the north (3')
#
# Compare like with like in time: take the old regions' orders from the day the north opened, so that the
# classifier cannot separate the two by season or year.
#
# ✏️ **Exercise** · Return the orders since `NORTH_START` with a column `north` (1 for the northern region, 0
# otherwise).

# %%
def two_samples(o: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    w = o[o.date >= NORTH_START].reset_index(drop=True)
    return w.assign(north=(w.region == "northern").astype(int))
    # END SOLUTION

# %%
try:
    w = two_samples(o)
except NotImplementedError:
    w = o[o.date >= NORTH_START].reset_index(drop=True)
    w = w.assign(north=(w.region == "northern").astype(int))
w = w.iloc[::U.pick(1, 3)].reset_index(drop=True)
print(w.groupby("north").agg(orders=("order", "size"), customers=("customer", "nunique")).to_string())
U.check(w.north.sum() > 0 and (w.north == 0).sum() > 0, "both samples have orders")

# %% [markdown]
# ## 2 · The adversarial classifier (8')
#
# Train a LightGBM to predict `north` from the model's own features (`U.xy`). Score it out of fold with folds
# grouped by customer: a customer is in one region, so a random split would let the classifier recognise
# customers instead of conditions.
#
# ✏️ **Exercise** · Return the out-of-fold probability of `north` for every order.

# %%
X, _ = U.xy(w)
y_adv = w.north.to_numpy()
folds = U.grouped_folds(w.customer, 5)


def adversarial_oof(X, y, folds) -> np.ndarray:
    # BEGIN SOLUTION
    p = np.zeros(len(y))
    for tr, te in folds:
        m = U.lgbm(n_estimators=200, n_jobs=1).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p
    # END SOLUTION

# %%
try:
    p_adv = adversarial_oof(X, y_adv, folds)
except NotImplementedError:
    p_adv = np.zeros(len(y_adv))
    for tr, te in folds:
        p_adv[te] = U.lgbm(n_estimators=200, n_jobs=1).fit(X.iloc[tr], y_adv[tr]).predict_proba(X.iloc[te])[:, 1]
adv_auc = U.auc(y_adv, p_adv)
print(f"adversarial AUC, grouped by customer: {adv_auc:.3f}  (0.5 = no shift the features can see)")
U.check(adv_auc > 0.8, "the classifier separates the north from the old depots")

# %% [markdown]
# **Read it.** An AUC near 0.5 says the features look alike; near 1 says they barely overlap. Overlap matters
# more than the number: a northern order with no look-alike among the old orders gets a prediction from a part of
# the feature space the model never saw.

# %% [markdown]
# ## 3 · Which features carry the shift (6')
#
# ✏️ **Exercise** · Fit the adversarial classifier on all rows and return its gain importance per feature, as
# shares that add up to 1, largest first. Then compare the medians of the top three features on both sides.

# %%
def shift_importance(X, y) -> pd.Series:
    # BEGIN SOLUTION
    m = U.lgbm(n_estimators=200, n_jobs=1).fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return (imp / imp.sum()).sort_values(ascending=False)
    # END SOLUTION

# %%
try:
    imp = shift_importance(X, y_adv)
except NotImplementedError:
    m = U.lgbm(n_estimators=200, n_jobs=1).fit(X, y_adv)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    imp = (imp / imp.sum()).sort_values(ascending=False)
print(imp.round(3).to_string())
top3 = list(imp.index[:3])
print("\nmedians by side:")
print(X.assign(north=y_adv).groupby("north")[top3].median().to_string())
print("\nsegment mix by side:")
print(pd.crosstab(w.north, w.segment, normalize="index").round(3).to_string())
U.check(np.isclose(imp.sum(), 1.0), "importance shares add up to 1")

# %% [markdown]
# **Look at.** Northern drivers are new (the region is a year old) and the customers are farther away. Those two
# facts, not the season, separate the samples.

# %% [markdown]
# ## 4 · Label and concept shift, checked apart (8')
#
# Covariate shift changes the features; **label shift** changes how often orders are late; **concept shift**
# changes the link between the features and lateness. Check each:
#
# - the late rate on each side;
# - a late model trained on the old depots, applied to the north: the late rate it predicts from the northern
#   features is what covariate shift alone would give; the rest of the gap is concept shift;
# - the late rate by distance band, on each side, for drivers with at least a year's tenure.
#
# ✏️ **Exercise** · Return the late rate of each side, and the mean predicted late rate in the north from a model
# trained on the old side.

# %%
def label_and_concept(w: pd.DataFrame, X: pd.DataFrame):
    # BEGIN SOLUTION
    rates = w.groupby("north").late.mean()
    old = (w.north == 0).to_numpy()
    m = U.lgbm(n_jobs=1).fit(X[old], w.late[old])
    predicted_north = float(m.predict_proba(X[~old])[:, 1].mean())
    return rates, predicted_north
    # END SOLUTION

# %%
try:
    rates, predicted_north = label_and_concept(w, X)
except NotImplementedError:
    rates = w.groupby("north").late.mean()
    old = (w.north == 0).to_numpy()
    predicted_north = float(U.lgbm(n_jobs=1).fit(X[old], w.late[old]).predict_proba(X[~old])[:, 1].mean())
print(f"late rate: old depots {rates[0]:.3f} | north {rates[1]:.3f}")
print(f"predicted for the north by the old model: {predicted_north:.3f}")
print(f"  covariate shift explains {predicted_north - rates[0]:+.3f}; the rest, {rates[1] - predicted_north:+.3f}, is a changed link")
bands = pd.cut(w.distance_km, [0, 15, 30, 60, 320], labels=["0-15", "15-30", "30-60", "60+"])
seasoned = w.driver_tenure_m >= 12
print("\nlate rate by distance band, drivers with 12 months or more:")
print(w[seasoned].assign(band=bands[seasoned]).pivot_table(index="band", columns="north", values="late", aggfunc=["mean", "size"], observed=True).round(3).to_string())
U.check(rates[1] > predicted_north > rates[0], "covariates explain part of the northern late rate, not all of it")
U.save_reference("d4_1", {"adv_auc": adv_auc, "importance": imp.round(4).to_dict(),
                          "medians": X.assign(north=y_adv).groupby("north")[top3].median().to_dict(),
                          "late_old": rates[0], "late_north": rates[1], "predicted_north": predicted_north})

# %% [markdown]
# **Look at.** The old model expects the north to be later than the old depots, because of the distances and the
# new drivers, but not as late as it is. At the same distance band, with seasoned drivers, the north is still
# later: the link between distance and lateness itself has changed (new routes, fewer roads). No reweighting of
# old data can teach a model that.

# %% [markdown]
# ## ✱ Stretch · Drop the top shifted feature
#
# Remove the feature with the largest shift importance from `X` and rerun Task 2. How far does the AUC fall?
# What does that say about removing shifted features as a fix?

# %% [markdown]
# ## Checkpoint
#
# 1. The adversarial AUC is 0.95. Does that mean the late model will fail in the north?
# 2. Why are the folds grouped by customer?
# 3. Which of the three shifts can importance weighting correct?
#
# <details><summary>Answers</summary>
#
# 1. Not by itself: it says the inputs differ. Whether the model fails is measured on northern outcomes (Lab 4.2).
# 2. Customers belong to one region; random folds would let the classifier recognise customers it has seen.
# 3. Covariate shift, where the link from features to outcome holds. Not concept shift.
#
# </details>
