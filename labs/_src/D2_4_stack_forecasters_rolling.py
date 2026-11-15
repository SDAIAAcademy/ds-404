# %% [markdown]
# # Lab 2.4 · Stacking on rolling-origin windows
#
# Day 2 · Session 4 (Ensembles worth their cost) · 25 min.
#
# DS-304 showed that a simple average of forecasts is hard to beat. Here you learn the weights instead: four
# fitted candidates, weights fitted only on windows that ended before the window they forecast, and the
# improvement over the best single model reported with a bootstrap interval over series. Running four
# models costs four times one: the gain has to be worth it.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/reference/benchmarks.npz` and `data/reference/zero_shot.npz` (six origins each:
# the four candidates are already fitted).
#
# **Tasks**
# - [ ] 1 · The four candidates and the windows (4')
# - [ ] 2 · Weights from earlier windows only (8')
# - [ ] 3 · Best single, average and stack on later windows (6')
# - [ ] 4 · The gain and its interval (7')
# - [ ] ✱ Stretch · Learn one weight set per series group

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
from scipy.optimize import minimize, nnls
import bustan as U

plt = U.style()

# %% [markdown]
# ## 1 · The four candidates and the windows (4')
#
# Six origins, 2025-10-05 to 2026-05-10. The weights used at origin *k* are fitted on origins 0 to *k*-1,
# whose 28 days all ended before origin *k*: no forecast is scored by weights that saw its answer.

# %%
S = U.series_subset()
Y = U.demand().iloc[:, S]
B, _ = U.benchmarks()
Z, who = U.zero_shot_supplied()
CANDS = {"seasonal naive": B[0][:, S], "smoothing": B[1][:, S], "LightGBM (tuned)": B[2][:, S],
         "zero-shot + covariates": Z[1][:, S]}
NAMES = list(CANDS)
F = np.stack([CANDS[n] for n in NAMES])                    # [model, origin, series, 3, h]
TEST = [2, 3, 4, 5]                                         # origins scored; each uses the ones before it
scales = np.array([[U.scale_of(Y[c].loc[:o].to_numpy()) for c in Y.columns] for o in U.ORIGINS])  # [origin, series]
actual = np.stack([U.actuals(Y, o) for o in U.ORIGINS])     # [origin, series, h]
print("candidates:", NAMES, "| zero-shot answered by:", who)
for k in TEST:
    print(f"origin {U.ORIGINS[k].date()}: weights from {[str(o.date()) for o in U.ORIGINS[:k]]}")

# %% [markdown]
# ## 2 · Weights from earlier windows only (8')
#
#
# Find weights *w* (non-negative, summing to one) that minimise the **same loss the stack will be judged on**:
# the mean scaled quantile loss of the combined forecast `sum_m w_m * forecast_m` over the training origins.
# The same weights combine all three quantiles. A softmax keeps the weights on the simplex, so a general
# minimiser (`scipy.optimize.minimize`, Nelder-Mead) can search freely.
#
# ✏️ **Exercise** · Return the weight vector fitted on the origins in `train` (a list of origin indices).

# %%
def sql_of(q, k):
    """Per-series scaled quantile loss of forecasts q [series, 3, h] at origin index k."""
    return U.scores(actual[k], q, scales[k])["sql"]


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def stack_weights(train: list) -> np.ndarray:
    # BEGIN SOLUTION
    def loss(v):
        w = softmax(v)
        return np.mean([np.nanmean(sql_of(np.tensordot(w, F[:, k], axes=1), k)) for k in train])
    r = minimize(loss, np.zeros(len(NAMES)), method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-5})
    return softmax(r.x)
    # END SOLUTION

# %%
def ref_stack_weights(train):
    def loss(v):
        w = softmax(v)
        return np.mean([np.nanmean(sql_of(np.tensordot(w, F[:, k], axes=1), k)) for k in train])
    return softmax(minimize(loss, np.zeros(len(NAMES)), method="Nelder-Mead",
                            options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-5}).x)


try:
    W = {k: stack_weights(list(range(k))) for k in TEST}
except NotImplementedError:
    stack_weights = ref_stack_weights
    W = {k: stack_weights(list(range(k))) for k in TEST}
print(pd.DataFrame(W, index=NAMES).rename(columns=lambda k: str(U.ORIGINS[k].date())).round(3).to_string())
U.check(all(np.isclose(w.sum(), 1) and (w >= 0).all() for w in W.values()), "weights are non-negative and sum to one")

# %% [markdown]
# **Look at.** How much do the weights move from one origin to the next? Stable weights are a sign that the
# stack learned something about the methods, not about one window.
#
# **The wrong loss.** The cell below fits the weights the common way instead: least squares on the medians
# (`scipy.optimize.nnls`). Compare its weights with yours.

# %%
def nnls_weights(train):
    A, b = [], []
    for k in train:
        s = scales[k][:, None]
        A.append((F[:, k, :, 1, :] / s[None]).reshape(len(NAMES), -1).T); b.append((actual[k] / s).ravel())
    A, b = np.vstack(A), np.concatenate(b)
    ok = ~np.isnan(b) & ~np.isnan(A).any(axis=1)
    w, _ = nnls(A[ok], b[ok])
    return w / w.sum()


W_ls = {k: nnls_weights(list(range(k))) for k in TEST}
print(pd.DataFrame(W_ls, index=NAMES).rename(columns=lambda k: str(U.ORIGINS[k].date())).round(3).to_string())

# %% [markdown]
# ## 3 · Best single, average and stack on later windows (6')
#
# The **best single** model must also be chosen from the earlier windows only: the model with the lowest mean
# quantile loss on origins before *k*.

# %%
rows = []
for k in TEST:
    past = {n: np.nanmean([np.nanmean(sql_of(F[m, j], j)) for j in range(k)]) for m, n in enumerate(NAMES)}
    best = min(past, key=past.get)
    stack = np.tensordot(W[k], F[:, k], axes=1)
    avg = F[:, k].mean(axis=0)
    stack_ls = np.tensordot(W_ls[k], F[:, k], axes=1)
    for name, q in (("best single: " + best, F[NAMES.index(best), k]), ("simple average", avg), ("stack", stack),
                    ("least-squares stack", stack_ls)):
        s = sql_of(q, k)
        rows.append(pd.DataFrame({"origin": U.ORIGINS[k], "series": Y.columns, "method": name.split(":")[0], "chosen": best, "sql": s}))
res = pd.concat(rows).dropna(subset=["sql"])
print(res.groupby("method").sql.mean().round(4).to_string())
print("best single chosen per origin:", res.drop_duplicates("origin").set_index("origin").chosen.to_dict())

# %% [markdown]
# ## 4 · The gain and its interval (7')
#
# ✏️ **Exercise** · Per series, average the quantile loss of the best single model and of the stack over the
# test origins, take the difference (best single minus stack: positive means the stack is better), and
# return the mean with a bootstrap interval over series (`U.bootstrap_ci`). Also return the relative gain.

# %%
def gain_with_interval(res: pd.DataFrame) -> dict:
    # BEGIN SOLUTION
    p = res.pivot_table(index="series", columns="method", values="sql", aggfunc="mean")
    d = p["best single"] - p["stack"]
    m, lo, hi = U.bootstrap_ci(d)
    return {"gain": m, "lo": lo, "hi": hi, "relative": m / p["best single"].mean()}
    # END SOLUTION

# %%
try:
    g = gain_with_interval(res)
except NotImplementedError:
    p = res.pivot_table(index="series", columns="method", values="sql", aggfunc="mean")
    d = p["best single"] - p["stack"]
    m, lo, hi = U.bootstrap_ci(d)
    g = {"gain": m, "lo": lo, "hi": hi, "relative": m / p["best single"].mean()}
print(f"stack against the best single model: gain {g['gain']:+.4f} ({g['relative']:+.1%}), 95% interval {g['lo']:+.4f} to {g['hi']:+.4f}")
verdict = ("the stack is worth running" if g["lo"] > 0 and g["relative"] >= 0.02 else
           "the gain is real but under 2%: weigh it against running four models" if g["lo"] > 0 else
           "the stack is worse: keep the single model" if g["hi"] < 0 else
           "the gain is inside the noise: keep the single model")
print("verdict:", verdict)
p2 = res.pivot_table(index="series", columns="method", values="sql", aggfunc="mean")
ga = U.bootstrap_ci(p2["best single"] - p2["simple average"])
gl = U.bootstrap_ci(p2["best single"] - p2["least-squares stack"])
print(f"simple average against the best single: {ga[0]:+.4f}, 95% interval {ga[1]:+.4f} to {ga[2]:+.4f}")
print(f"least-squares stack against the best single: {gl[0]:+.4f}, 95% interval {gl[1]:+.4f} to {gl[2]:+.4f}")
U.check(g["lo"] <= g["gain"] <= g["hi"], "the gain sits inside its interval")
U.save_reference("d2_4", {"answered_by": who, "weights": {str(U.ORIGINS[k].date()): dict(zip(NAMES, W[k])) for k in TEST},
                          "means": res.groupby("method").sql.mean().to_dict(), "gain": g, "average_gain": ga, "ls_gain": gl, "verdict": verdict,
                          "weights_ls": {str(U.ORIGINS[k].date()): dict(zip(NAMES, W_ls[k])) for k in TEST}})

# %% [markdown]
# **The cost line.** The stack needs all four models every week: four fits, four forecast runs, four things
# to monitor. Write the gain next to that cost before recommending it.

# %% [markdown]
# ## ✱ Stretch · Weights per group
#
# Fit one weight vector per series group (Lab 2.2's groups) on the same earlier windows. Does the group-wise
# stack beat the global one by more than its interval, or does it overfit the small groups?

# %% [markdown]
# ## Checkpoint
#
# 1. Why are the weights for origin 2026-03-15 fitted on the four earlier origins only?
# 2. The stack beats the best single model by 2% with an interval from -0.5% to 4.5%. What do you recommend?
# 3. Why choose the "best single model" on earlier windows too?
#
# <details><summary>Answers</summary>
#
# 1. Their windows end on or before 2026-03-15 (2026-02-15's reaches it exactly); a later window would hold the
#    days being scored.
# 2. Keep the single model, or rerun with more origins: the gain is not distinguishable from zero and the stack
#    costs four models.
# 3. Picking it on the test windows gives the single model an advantage the stack does not get.
#
# </details>
#
# **Carry forward.** The weights and the gain are in `data/reference/d2_4.json`.