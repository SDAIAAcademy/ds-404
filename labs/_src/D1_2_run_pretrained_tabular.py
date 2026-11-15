# %% [markdown]
# # Lab 1.2 · A pretrained model on the same folds
#
# Day 1 · Session 2 (In-context learning on tables) · 25 min.
#
# A tabular foundation model was pretrained on a very large number of synthetic tables. It reads Bustan's training
# orders as context and predicts the late flag without being trained on them. You run it on exactly the
# folds and the metric of the baseline, time it, and build a comparison table that counts the baseline's
# tuning time too.
#
# **Where to run it.** In Google Colab the setup cell installs `tabicl` and its checkpoint downloads into the
# Colab runtime (a GPU runtime is faster; a CPU runtime works for 2,000 rows). Nothing downloads to your
# laptop. Outside Colab a **stand-in** answers: a spline logistic regression averaged with a nearest-neighbour
# vote, which, like an in-context learner, does no tuning. Every result says which one answered.
#
# **AI policy: AI-tutor** (hints, not finished code).
#
# **Reference state.** `data/orders.csv.gz` and `data/reference/d1_1.json` (the baseline configuration from
# Lab 1.1, supplied: you do not need your own).
#
# **Tasks**
# - [ ] 1 · Who answers: the model or the stand-in (3')
# - [ ] 2 · Same folds, same metric (6')
# - [ ] 3 · Fit and predict, timed (7')
# - [ ] 4 · The comparison table with tuning time (6')
# - [ ] 5 · What changed in the preprocessing (3')
# - [ ] ✱ Stretch · Give the pretrained model the raw categories and compare

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
# ## 1 · Who answers (3')
#
# `U.tabular_model()` returns the pretrained model when the runtime has it, the stand-in otherwise, and says
# which. Read the line it prints before you read any number below.

# %%
model, info = U.tabular_model()
print("answered by:", info["model"])
if not info["live"]:
    print("!! STAND-IN: not a pretrained model. Run this notebook in Colab for TabICL's own numbers.")

pool, _ = U.curve_split()
N = U.pick(2000, 1000)
data = pool.sample(N, random_state=U.SEED + 12).reset_index(drop=True)
X, y = U.xy(data)
base = U.reference("d1_1")
best_cfg = base["best_config"]
print(f"{N:,} orders, late rate {y.mean():.3f}; baseline configuration from Lab 1.1: {best_cfg}")

# %% [markdown]
# ## 2 · Same folds, same metric (6')
#
# ✏️ **Exercise** · Score any model on given folds. `make_model()` returns a fresh, unfitted model with
# `fit` and `predict_proba`. Return a table with one row per fold: `auc`, `fit_s`, `predict_s`.

# %%
def fold_scores(make_model, X, y, folds) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for tr, te in folds:
        m = make_model()
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y[tr])
        t1 = time.perf_counter()
        p = m.predict_proba(X.iloc[te])[:, 1]
        t2 = time.perf_counter()
        rows.append({"auc": U.auc(y[te], p), "fit_s": t1 - t0, "predict_s": t2 - t1})
    return pd.DataFrame(rows)
    # END SOLUTION

# %%
def ref_fold_scores(make_model, X, y, folds):
    rows = []
    for tr, te in folds:
        m = make_model(); t0 = time.perf_counter(); m.fit(X.iloc[tr], y[tr]); t1 = time.perf_counter()
        p = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({"auc": U.auc(y[te], p), "fit_s": t1 - t0, "predict_s": time.perf_counter() - t1})
    return pd.DataFrame(rows)


folds = U.grouped_folds(data.customer, U.pick(5, 3))
try:
    gbm = fold_scores(lambda: U.lgbm(**best_cfg), X, y, folds)
except NotImplementedError:
    fold_scores = ref_fold_scores
    gbm = fold_scores(lambda: U.lgbm(**best_cfg), X, y, folds)
U.check(len(gbm) == len(folds) and gbm.auc.between(0.5, 1).all(), "one baseline AUC per fold")
print(gbm.round(3).to_string())

# %% [markdown]
# ## 3 · Fit and predict, timed (7')
#
# Run the pretrained model (or the stand-in) on the same folds with `fold_scores`. Then compute the paired
# difference per fold (pretrained minus baseline) and its interval with `U.mean_ci`.

# %%
def make_pretrained():
    return U.tabular_model()[0]


pre = fold_scores(make_pretrained, X, y, folds)
print(pre.round(3).to_string())
diff = pre.auc.to_numpy() - gbm.auc.to_numpy()
m, lo, hi = U.mean_ci(diff)
print(f"paired AUC difference (pretrained minus baseline): {m:+.4f}, 95% interval {lo:+.4f} to {hi:+.4f}")
U.check(len(pre) == len(gbm), "both models scored on the same folds")

# %% [markdown]
# **Look at.** Where does the pretrained model spend its time: in `fit` or in `predict`? An in-context
# learner has almost nothing to fit: the training rows are read at prediction time, so its cost grows with
# training rows **times** test rows.

# %% [markdown]
# ## 4 · The comparison table with tuning time (6')
#
# The baseline's AUC was bought with a search: Lab 1.1 spent `fits` model fits. The pretrained model ran
# with its defaults.
#
# ✏️ **Exercise** · Return a table with one row per model and the columns `answered_by`, `auc`, `auc_sd`,
# `fit_s`, `predict_s`, `tuning_s`. For the baseline, `tuning_s` is the number of fits of Lab 1.1 times its
# mean fit time here; for the pretrained model it is 0.

# %%
def comparison(gbm: pd.DataFrame, pre: pd.DataFrame, fits: int, who: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for name, d, tune in (("tuned LightGBM", gbm, fits * gbm.fit_s.mean()), (who, pre, 0.0)):
        rows.append({"answered_by": name, "auc": d.auc.mean(), "auc_sd": d.auc.std(ddof=1), "fit_s": d.fit_s.mean(),
                     "predict_s": d.predict_s.mean(), "tuning_s": tune})
    return pd.DataFrame(rows)
    # END SOLUTION

# %%
try:
    table = comparison(gbm, pre, base["fits"], info["model"])
except NotImplementedError:
    table = pd.DataFrame([{"answered_by": n, "auc": d.auc.mean(), "auc_sd": d.auc.std(ddof=1), "fit_s": d.fit_s.mean(),
                           "predict_s": d.predict_s.mean(), "tuning_s": t}
                          for n, d, t in (("tuned LightGBM", gbm, base["fits"] * gbm.fit_s.mean()), (info["model"], pre, 0.0))])
print(table.round(4).to_string(index=False))
U.check(table.tuning_s.iloc[0] > table.fit_s.iloc[0], "the baseline's tuning costs more than one fit")
U.save_reference("d1_2", {"n": N, "folds": len(folds), "answered_by": info["model"], "table": table.to_dict("records"),
                          "diff": [m, lo, hi]})

# %% [markdown]
# ## 5 · What changed in the preprocessing (3')
#
# The boosted trees took the numbers as they are: a tree only needs an order. A pretrained tabular model
# brings its own preprocessing: it rescales each column, encodes categories and averages several passes over
# transformed copies of the table (TabICL's `n_estimators`). Read the dtypes below and answer: which column would a careless pipeline feed as a number
# when it is a category?

# %%
print(X.dtypes.to_string())
print("segment codes:", dict(enumerate(U.SEGMENTS)))

# %% [markdown]
# ## ✱ Stretch · Raw categories
#
# Pass `segment` as text (`data.segment`) to the pretrained model instead of the code. TabICL and the
# Does the model accept it (read its documentation first), and does the AUC move? The stand-in one-hot encodes the
# code already.

# %% [markdown]
# ## Checkpoint
#
# 1. The pretrained model's `fit` takes a fraction of a second and its `predict` takes longer. Why?
# 2. Your table shows the two AUCs within 0.01 and the interval of the difference crosses zero. What may you
#    claim?
# 3. Why must both models use the same folds rather than each its own random split?
#
# <details><summary>Answers</summary>
#
# 1. An in-context learner does its work at prediction time: it attends from each test row to the training rows.
# 2. Only that this experiment cannot tell them apart at 2,000 orders; Lab 1.3 asks the same question at
#    other sizes.
# 3. Paired folds remove the split-to-split noise from the difference; separate splits add it back.
#
# </details>
#
# **Carry forward.** The comparison table is in `data/reference/d1_2.json`.
