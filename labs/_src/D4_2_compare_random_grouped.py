# %% [markdown]
# # Lab 4.2 · Four schemes against deployed error
#
# Day 4 · Session 2 (Validating under shift) · 25 min.
#
# Before the north opened, an analyst validated the late-delivery model on the old depots and promised Shadi an
# AUC. You can now check that promise: validate the same model four ways on the old depots only, then measure the
# error it actually had on 8,000 northern orders. The honest scheme is the one whose estimate came closest.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/orders.csv.gz`; the model is a LightGBM with fixed settings (`U.lgbm`), the same in
# every scheme.
#
# **Tasks**
# - [ ] 1 · One model, four schemes (8')
# - [ ] 2 · Deployed error in the north (5')
# - [ ] 3 · Which scheme was honest (5')
# - [ ] 4 · Error by region and segment (7')
# - [ ] ✱ Stretch · Validate on Hail only and predict Tabuk

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
o = U.orders()
old = o[o.region != "northern"].iloc[::U.pick(1, 3)].reset_index(drop=True)
north = o[o.region == "northern"].reset_index(drop=True)
Xo, yo = U.xy(old)
Xn, yn = U.xy(north)
MODEL = lambda: U.lgbm(n_estimators=200, n_jobs=1)   # one thread: on some Windows laptops four are far slower
print(f"old depots: {len(old):,} orders, {old.customer.nunique():,} customers | north: {len(north):,} orders")


def score(y, p) -> dict:
    return {"auc": U.auc(y, p), "logloss": U.logloss(y, p), "predicted": float(np.mean(p)), "observed": float(np.mean(y))}

# %% [markdown]
# ## 1 · One model, four schemes (8')
#
# On the old depots only:
#
# - **random**: 5 folds, orders shuffled;
# - **grouped**: 5 folds, each customer on one side;
# - **temporal**: train on the first 80% of days, test on the last 20%;
# - **region-held-out**: train on two old regions, test on the third, three times.
#
# ✏️ **Exercise** · Return, for each scheme, the list of `(train_index, test_index)` pairs.

# %%
def schemes(old: pd.DataFrame) -> dict:
    # BEGIN SOLUTION
    cut = old.date.quantile(0.8)
    return {
        "random": U.random_folds(len(old), 5),
        "grouped": U.grouped_folds(old.customer, 5),
        "temporal": [(np.flatnonzero(old.date < cut), np.flatnonzero(old.date >= cut))],
        "region-held-out": [(np.flatnonzero(old.region != r), np.flatnonzero(old.region == r))
                            for r in ["central", "western", "eastern"]],
    }
    # END SOLUTION

# %%
def ref_schemes(old):
    cut = old.date.quantile(0.8)
    return {"random": U.random_folds(len(old), 5), "grouped": U.grouped_folds(old.customer, 5),
            "temporal": [(np.flatnonzero(old.date < cut), np.flatnonzero(old.date >= cut))],
            "region-held-out": [(np.flatnonzero(old.region != r), np.flatnonzero(old.region == r)) for r in ["central", "western", "eastern"]]}


try:
    S = schemes(old)
except NotImplementedError:
    S = ref_schemes(old)
U.check(set(S) == {"random", "grouped", "temporal", "region-held-out"}, "four schemes")
rows, oof = [], np.zeros(len(old))
for name, folds in S.items():
    s = []
    for tr, te in folds:
        p = MODEL().fit(Xo.iloc[tr], yo[tr]).predict_proba(Xo.iloc[te])[:, 1]
        s.append(score(yo[te], p))
        if name == "grouped":
            oof[te] = p                                  # kept for Task 4
    rows.append({"scheme": name, **pd.DataFrame(s).mean().to_dict()})
est = pd.DataFrame(rows).set_index("scheme")
print(est.round(3).to_string())

# %% [markdown]
# ## 2 · Deployed error in the north (5')
#
# ✏️ **Exercise** · Train the model on all old-depot orders, predict the northern orders and return their scores
# with `score`. Keep the predictions in `p_north`.

# %%
def deployed(Xo, yo, Xn, yn):
    # BEGIN SOLUTION
    p = MODEL().fit(Xo, yo).predict_proba(Xn)[:, 1]
    return score(yn, p), p
    # END SOLUTION

# %%
try:
    dep, p_north = deployed(Xo, yo, Xn, yn)
except NotImplementedError:
    p_north = MODEL().fit(Xo, yo).predict_proba(Xn)[:, 1]
    dep = score(yn, p_north)
print({k: round(v, 3) for k, v in dep.items()})
U.check(dep["observed"] > dep["predicted"], "the model under-predicts lateness in the north")

# %% [markdown]
# ## 3 · Which scheme was honest (5')
#
# ✏️ **Exercise** · Add the columns `auc_gap` and `logloss_gap` (estimate minus deployed) to `est`, and name the
# scheme whose AUC estimate was closest to the deployed AUC.

# %%
def honesty(est: pd.DataFrame, dep: dict):
    # BEGIN SOLUTION
    t = est.assign(auc_gap=est.auc - dep["auc"], logloss_gap=est.logloss - dep["logloss"])
    return t, t.auc_gap.abs().idxmin()
    # END SOLUTION

# %%
try:
    est, closest = honesty(est, dep)
except NotImplementedError:
    est = est.assign(auc_gap=est.auc - dep["auc"], logloss_gap=est.logloss - dep["logloss"])
    closest = est.auc_gap.abs().idxmin()
print(est.round(3).to_string())
print(f"\nclosest AUC estimate: {closest}")
U.check(est.loc["random", "auc_gap"] > est.loc["grouped", "auc_gap"], "random folds flatter the model more than grouped folds")

# %% [markdown]
# **Look at.** Two things. The random and the temporal schemes both let a customer's orders sit on both sides:
# they promise an AUC the model never had for new customers. And every scheme, even the closest on AUC,
# promised a log loss far below the north's: ranking survived the move, calibration did not, because the late
# rate and the link to distance changed (Lab 4.1). No split of old data can show a shift that only exists in the
# new region.

# %% [markdown]
# ## 4 · Error by region and segment (7')
#
# ✏️ **Exercise** · Return a table of AUC, predicted and observed late rate per segment in the north, from
# `p_north`. The grouped scheme's per-region scores on the old depots are printed for comparison.

# %%
def by_segment(north: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    # BEGIN SOLUTION
    t = north.assign(p=p)
    return pd.DataFrame({s: {"orders": len(g), "auc": U.auc(g.late, g.p), "predicted": g.p.mean(), "observed": g.late.mean()}
                         for s, g in t.groupby("segment")}).T
    # END SOLUTION

# %%
try:
    seg = by_segment(north, p_north)
except NotImplementedError:
    t = north.assign(p=p_north)
    seg = pd.DataFrame({s: {"orders": len(g), "auc": U.auc(g.late, g.p), "predicted": g.p.mean(), "observed": g.late.mean()} for s, g in t.groupby("segment")}).T
print("north, deployed, by segment:")
print(seg.round(3).to_string())
reg = pd.DataFrame({r: {"orders": len(g), "auc": U.auc(g.late, g.p), "predicted": g.p.mean(), "observed": g.late.mean()}
                    for r, g in old.assign(p=oof).groupby("region")}).T
print("\nold depots, grouped scheme, by region:")
print(reg.round(3).to_string())
worst = seg.auc.idxmin()
print(f"\nweakest northern segment: {worst} (AUC {seg.auc.min():.3f}, observed {seg.loc[worst, 'observed']:.3f} against predicted {seg.loc[worst, 'predicted']:.3f})")
U.check(len(seg) == 4, "four segments scored")
U.save_reference("d4_2", {"estimates": est.reset_index().to_dict("records"), "deployed": dep, "closest": closest,
                          "by_segment": seg.reset_index(names="segment").to_dict("records"),
                          "by_region_old": reg.reset_index(names="region").to_dict("records")})

# %% [markdown]
# ## ✱ Stretch · Hail to Tabuk
#
# Train on the old depots and Hail (`depot == "HAS"`), test on Tabuk (`"TUU"`). Does adding a few months of the
# new region fix the calibration? Compare the predicted and observed late rate in Tabuk.

# %% [markdown]
# ## Checkpoint
#
# 1. Which two schemes leaked, and how?
# 2. Region-held-out came closest on AUC. Why did it still miss the log loss?
# 3. What do you promise Shadi before the north has outcomes?
#
# <details><summary>Answers</summary>
#
# 1. Random and temporal: the same customers' orders were in training and test.
# 2. The old regions resemble each other; the north differs in its late rate and its distance effect, which no
#    old region shows.
# 3. The region-held-out AUC as an upper estimate of ranking, no calibrated probabilities until northern outcomes
#    are in, and a date to recalibrate.
#
# </details>
