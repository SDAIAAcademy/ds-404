# %% [markdown]
# # Lab 1.1 · A tuned baseline under a budget
#
# Day 1 · Session 1 (Tuned boosted trees first) · 25 min.
#
# Before any pretrained model is tested, Bustan needs a baseline that was given a fair chance: a LightGBM
# model tuned by random search under a stated budget, validated with folds that keep each customer on one
# side. You fix the budget, write the grouped folds and the search, see how much random K-fold flatters the
# model, and average the three best configurations.
#
# **AI policy: AI-off.** This baseline is written without an assistant: it is the yardstick of the day.
#
# **Reference state.** `data/orders.csv.gz` (the established regions only). Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Load the orders and fix the budget (3')
# - [ ] 2 · Grouped folds by customer (5')
# - [ ] 3 · Random search under the budget (8')
# - [ ] 4 · Grouped against random K-fold (5')
# - [ ] 5 · Ensemble the top three (4')
# - [ ] ✱ Stretch · Spend the same budget on a finer grid of two parameters

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

import time
import numpy as np
import pandas as pd
import bustan as U

plt = U.style()

# %% [markdown]
# ## 1 · Load the orders and fix the budget (3')
#
# The baseline is trained on 5,000 orders of the established regions and judged on a fixed set of held-out
# customers (`U.curve_split()`): the same test set Lab 1.3 uses. The budget is written down **before** the
# search: a number of configurations, and the fits it costs.

# %%
pool, test = U.curve_split()
N = U.pick(5000, 2000)
train = pool.sample(N, random_state=U.SEED).reset_index(drop=True)
X, y = U.xy(train)
Xt, yt = U.xy(test)
BUDGET = U.pick(20, 6)           # configurations
K = U.pick(5, 3)                 # folds
print(f"{len(train):,} training orders from {train.customer.nunique():,} customers; late rate {y.mean():.3f}")
print(f"held-out test: {len(test):,} orders from {test.customer.nunique():,} other customers")
print(f"budget: {BUDGET} configurations x {K} folds = {BUDGET * K} fits, plus 3 refits")

# %% [markdown]
# **Look at.** How many orders does a customer place in this sample? If a customer appears in both the
# training part and the validation part of a fold, the model can recognise them by their fixed distance and
# segment: that is the flattery task 4 measures.

# %%
per_cust = train.customer.value_counts()
print("orders per customer: median", int(per_cust.median()), "| max", int(per_cust.max()),
      "| customers with 3 or more:", int((per_cust >= 3).sum()))

# %% [markdown]
# ## 2 · Grouped folds by customer (5')
#
# ✏️ **Exercise** · Return `k` folds as a list of `(train_index, validation_index)` arrays such that every
# customer's orders fall in one fold only. Shuffle the customers with `np.random.default_rng(seed)`.

# %%
def customer_folds(customers: pd.Series, k: int = 5, seed: int = 404) -> list:
    # BEGIN SOLUTION
    uniq = np.sort(customers.unique())
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % k))
    f = customers.map(fold_of).to_numpy()
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    folds = customer_folds(train.customer, K)
except NotImplementedError:
    folds = U.grouped_folds(train.customer, K)
U.check(len(folds) == K, f"{K} folds")
U.check(all(not set(train.customer.iloc[a]) & set(train.customer.iloc[b]) for a, b in folds),
        "no customer is on both sides of any fold")
U.check(sum(len(b) for _, b in folds) == len(train), "every order is validated exactly once")

# %% [markdown]
# ## 3 · Random search under the budget (8')
#
# `U.sample_config(rng)` draws one configuration from the search space (learning rate, leaves, minimum rows
# per leaf, trees, feature share, L2 penalty). `U.lgbm(**config)` builds the model.
#
# ✏️ **Exercise** · Draw `budget` configurations with one generator, score each by its mean validation AUC
# over the folds (`U.auc`), and return a table with columns `config`, `auc`, `seconds`, best first.

# %%
def random_search(X, y, folds, budget: int, seed: int = 404) -> pd.DataFrame:
    # BEGIN SOLUTION
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(budget):
        cfg = U.sample_config(rng)
        t0 = time.perf_counter()
        s = [U.auc(y[b], U.lgbm(**cfg).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]) for a, b in folds]
        rows.append({"config": cfg, "auc": float(np.mean(s)), "seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    # END SOLUTION

# %%
t0 = time.perf_counter()
try:
    search = random_search(X, y, folds, BUDGET)
except NotImplementedError:
    rng = np.random.default_rng(404)
    cfgs = [U.sample_config(rng) for _ in range(BUDGET)]
    search = pd.DataFrame({"config": cfgs, "auc": [U.cv_auc(lambda c=c: U.lgbm(**c), X, y, folds) for c in cfgs],
                           "seconds": np.nan}).sort_values("auc", ascending=False).reset_index(drop=True)
search_seconds = time.perf_counter() - t0
U.check(len(search) == BUDGET and search.auc.is_monotonic_decreasing, "one row per configuration, best first")
print(f"search: {BUDGET * K} fits in {search_seconds:.0f} s")
print(search.assign(auc=search.auc.round(4), seconds=search.seconds.round(1)).head(5).to_string())
best_cfg = search.config[0]
print("spread between best and worst configuration:", round(search.auc.max() - search.auc.min(), 4))

# %% [markdown]
# ## 4 · Grouped against random K-fold (5')
#
# Score the best configuration with ordinary random folds (`U.random_folds(len(train), K)`) and compare
# both estimates with the score on the held-out customers.
#
# ✏️ **Exercise** · Return the three AUC values: grouped CV, random CV, held-out test.

# %%
def three_estimates(cfg, X, y, folds, Xt, yt) -> dict:
    # BEGIN SOLUTION
    grouped = U.cv_auc(lambda: U.lgbm(**cfg), X, y, folds)
    rand = U.cv_auc(lambda: U.lgbm(**cfg), X, y, U.random_folds(len(y), len(folds)))
    held = U.auc(yt, U.lgbm(**cfg).fit(X, y).predict_proba(Xt)[:, 1])
    return {"grouped_cv": grouped, "random_cv": rand, "held_out": held}
    # END SOLUTION

# %%
try:
    est = three_estimates(best_cfg, X, y, folds, Xt, yt)
except NotImplementedError:
    est = {"grouped_cv": U.cv_auc(lambda: U.lgbm(**best_cfg), X, y, folds),
           "random_cv": U.cv_auc(lambda: U.lgbm(**best_cfg), X, y, U.random_folds(len(y), K)),
           "held_out": U.auc(yt, U.lgbm(**best_cfg).fit(X, y).predict_proba(Xt)[:, 1])}
for k_, v in est.items():
    print(f"{k_:>11}: {v:.4f}")
gap_random = est["random_cv"] - est["held_out"]
gap_grouped = est["grouped_cv"] - est["held_out"]
print(f"random K-fold is {gap_random:+.4f} above the held-out customers; grouped is {gap_grouped:+.4f}")
U.check(est["random_cv"] > est["grouped_cv"], "random K-fold scores the same model higher than grouped folds")

# %% [markdown]
# ## 5 · Ensemble the top three (4')
#
# ✏️ **Exercise** · Refit the three best configurations on all training orders and average their predicted
# probabilities on the held-out test. Return the averaged probabilities.

# %%
def top3_ensemble(search: pd.DataFrame, X, y, Xt) -> np.ndarray:
    # BEGIN SOLUTION
    return np.mean([U.lgbm(**c).fit(X, y).predict_proba(Xt)[:, 1] for c in search.config[:3]], axis=0)
    # END SOLUTION

# %%
try:
    p_ens = top3_ensemble(search, X, y, Xt)
except NotImplementedError:
    p_ens = np.mean([U.lgbm(**c).fit(X, y).predict_proba(Xt)[:, 1] for c in search.config[:3]], axis=0)
ens_auc = U.auc(yt, p_ens)
print(f"best single on held-out: {est['held_out']:.4f} | top-three ensemble: {ens_auc:.4f}")
U.check(ens_auc > 0.70, "the ensemble ranks late orders well above chance")

summary = {"n_train": N, "budget": BUDGET, "folds": K, "fits": BUDGET * K + 3, "search_seconds": search_seconds,
           "best_config": best_cfg, **est, "ensemble_auc": ens_auc}
U.save_reference("d1_1", summary)

# %% [markdown]
# ## ✱ Stretch · A finer grid of two parameters
#
# Spend the same number of fits on a grid of `learning_rate` x `num_leaves` (for example 4 x 5 values at
# five folds) with the other parameters at the best configuration's values. Does a grid beat random search
# at the same budget? Bergstra and Bengio's argument says random search wins when only a few parameters
# matter: check which parameters mattered here.

# %% [markdown]
# ## Checkpoint
#
# 1. A colleague reports 0.84 AUC from random 5-fold CV on the same orders. What do you tell him?
# 2. Your search took 100 fits. A pretrained model will be run once. Which numbers must the comparison table
#    carry so that it is fair?
# 3. Why are the three configurations averaged on probabilities, not on predicted classes?
#
# <details><summary>Answers</summary>
#
# 1. Random folds put a customer's orders on both sides, and the model recognises customers by fixed fields
#    (distance, segment). New customers are what Bustan predicts for: use grouped folds or held-out customers.
# 2. The tuning budget in fits and seconds, the fit and predict time of each model, and the same folds and
#    metric for both.
# 3. AUC and the cost decisions of Day 3 use the ranking and the probability; averaging classes throws both away.
#
# </details>
#
# **Carry forward.** The best configuration and the three estimates are in `data/reference/d1_1.json`;
# Lab 1.2 starts from that file.
