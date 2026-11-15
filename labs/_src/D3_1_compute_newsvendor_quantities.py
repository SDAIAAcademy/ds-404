# %% [markdown]
# # Lab 3.1 · Newsvendor quantities
#
# Day 3 · Session 1 (Asymmetric costs) · 25 min.
#
# Imad Al-Sharqi orders each week for the Riyadh depot. A case short costs Bustan its margin and some of the
# customer's goodwill; a case left over costs holding, and for chilled dairy most of its value. You turn the
# cost table into a critical ratio per product, turn daily quantile forecasts into weekly ones, order the
# matching quantile for 20 products, and measure what ordering the median would cost instead.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/products.csv` and the supplied zero-shot forecasts with covariates
# (`data/reference/zero_shot.npz`, origin 2026-05-10): the week of 11 to 17 May 2026.
#
# **Tasks**
# - [ ] 1 · Underage and overage from the cost table (5')
# - [ ] 2 · The critical ratio (4')
# - [ ] 3 · Weekly quantiles from daily ones (6')
# - [ ] 4 · Order quantities for 20 products (5')
# - [ ] 5 · What ordering the median costs (5')
# - [ ] ✱ Stretch · Change the shortage cost of dates and watch the quantity move

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
from scipy import stats
import bustan as U

plt = U.style()
pd.set_option("display.width", 160)

# %% [markdown]
# ## 1 · Underage and overage from the cost table (5')
#
# Per case, for one week:
#
# - **underage** (a case short): the lost margin `price - unit_cost`, plus goodwill worth half the margin;
# - **overage** (a case left over): holding at 0.4% of the unit cost per day for seven days, plus, for
#   products with a shelf life of 30 days or less, 60% of the unit cost lost to spoilage.
#
# ✏️ **Exercise** · Add the columns `underage` and `overage` to the 20 products.

# %%
p = U.day3_products().drop(columns=["underage", "overage"])


def add_costs(p: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    margin = p.price - p.unit_cost
    p = p.copy()
    p["underage"] = (1.5 * margin).round(2)
    p["overage"] = (p.unit_cost * 0.004 * 7 + np.where(p.shelf_days <= 30, 0.6 * p.unit_cost, 0.0)).round(2)
    return p
    # END SOLUTION

# %%
ref = U.day3_products()
try:
    p = add_costs(p)
except NotImplementedError:
    p = ref.copy()
U.check(np.allclose(p.underage, ref.underage) and np.allclose(p.overage, ref.overage), "costs match the reference table")
print(p[["product", "category", "supplier", "unit_cost", "price", "shelf_days", "underage", "overage"]].to_string(index=False))

# %% [markdown]
# ## 2 · The critical ratio (4')
#
# Order up to the demand quantile at **underage / (underage + overage)**. When a shortage costs nine times a
# leftover, that is the 0.9 quantile.
#
# ✏️ **Exercise** · Add the column `cr`.

# %%
def add_cr(p: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    return p.assign(cr=p.underage / (p.underage + p.overage))
    # END SOLUTION

# %%
try:
    p = add_cr(p)
except NotImplementedError:
    p = p.assign(cr=p.underage / (p.underage + p.overage))
print(p.groupby("category").cr.agg(["min", "max"]).round(2).to_string())
U.check(p.cr.between(0, 1).all(), "every critical ratio is a probability")

# %% [markdown]
# **Look at.** Dry goods sit near 0.9; chilled dairy below 0.35. Same forecaster, opposite decisions: dairy is
# ordered **below** its median because a leftover is mostly lost.

# %% [markdown]
# ## 3 · Weekly quantiles from daily ones (6')
#
# The forecaster gives the 0.1, 0.5 and 0.9 quantiles of each day. Imad orders for the week. Approximate the
# weekly total as normal: its mean is the sum of the daily medians, and each day's spread is its 0.1 to 0.9
# range divided by 2.563 (the width of that range in standard deviations). Assume the days are independent:
# Lab 3.4 tests that assumption.
#
# ✏️ **Exercise** · Return `mu` and `sd` of the weekly total for each product from `q [products, 3, 7]`.

# %%
q = U.daily_forecasts("zero_shot_cov")[:, :, :7]
week = pd.date_range("2026-05-11", periods=7, freq="D")


def weekly_normal(q: np.ndarray):
    # BEGIN SOLUTION
    mu = q[:, 1].sum(axis=1)
    sd = np.sqrt((((q[:, 2] - q[:, 0]) / 2.563) ** 2).sum(axis=1))
    return mu, sd
    # END SOLUTION

# %%
try:
    mu, sd = weekly_normal(q)
except NotImplementedError:
    mu, sd = q[:, 1].sum(axis=1), np.sqrt((((q[:, 2] - q[:, 0]) / 2.563) ** 2).sum(axis=1))
wi = U.week_inputs()
U.check(np.allclose(mu, wi.mu) and np.allclose(sd, wi.sd), "weekly mean and spread match the reference")
print(f"week {week[0].date()} to {week[-1].date()}; weekly demand of P001: mean {mu[0]:.0f} cases, sd {sd[0]:.0f}")

# %% [markdown]
# ## 4 · Order quantities for 20 products (5')
#
# ✏️ **Exercise** · The newsvendor quantity is the weekly quantile at the critical ratio, rounded **up** to whole
# cases and never below zero. Add the column `order`.

# %%
def newsvendor(p: pd.DataFrame, mu, sd) -> pd.DataFrame:
    # BEGIN SOLUTION
    return p.assign(mu=mu, sd=sd, order=np.ceil(np.maximum(mu + sd * stats.norm.ppf(p.cr), 0)))
    # END SOLUTION

# %%
try:
    p = newsvendor(p, mu, sd)
except NotImplementedError:
    p = p.assign(mu=mu, sd=sd, order=np.ceil(np.maximum(mu + sd * stats.norm.ppf(p.cr), 0)))
U.check(np.allclose(p.order, wi.target), "order quantities match the reference")
print(p[["product", "category", "cr", "mu", "sd", "order"]].round(2).to_string(index=False))

# %% [markdown]
# ## 5 · What ordering the median costs (5')
#
# Simulate 20,000 weeks of demand per product from the same normal (clipped at zero) and compare the
# expected cost of the newsvendor order with ordering the median (`mu`, rounded up).
#
# ✏️ **Exercise** · Return the expected cost per product of an order vector, from simulated demand `D`
# `[sims, products]`: underage times the shortfall plus overage times the leftover, averaged over simulations.

# %%
rng = np.random.default_rng(U.SEED)
D = np.maximum(rng.normal(p.mu.to_numpy(), p.sd.to_numpy(), (20000, len(p))), 0)


def expected_cost(order, D, under, over) -> np.ndarray:
    # BEGIN SOLUTION
    short = np.maximum(D - order, 0)
    left = np.maximum(order - D, 0)
    return (under * short + over * left).mean(axis=0)
    # END SOLUTION

# %%
u, o = p.underage.to_numpy(), p.overage.to_numpy()
try:
    c_nv = expected_cost(p.order.to_numpy(), D, u, o)
except NotImplementedError:
    expected_cost = lambda order, D, under, over: (under * np.maximum(D - order, 0) + over * np.maximum(order - D, 0)).mean(axis=0)
    c_nv = expected_cost(p.order.to_numpy(), D, u, o)
c_med = expected_cost(np.ceil(p.mu.to_numpy()), D, u, o)
p["cost_newsvendor"], p["cost_median"] = c_nv.round(0), c_med.round(0)
print(p[["product", "category", "cr", "order", "cost_newsvendor", "cost_median"]].to_string(index=False))
saving = c_med.sum() - c_nv.sum()
print(f"expected weekly cost, 20 products: newsvendor SAR {c_nv.sum():,.0f} | median SAR {c_med.sum():,.0f} | saving SAR {saving:,.0f} ({saving / c_med.sum():.0%})")
U.check(c_nv.sum() < c_med.sum(), "ordering the critical quantile costs less than ordering the median")
U.save_reference("d3_1", {"products": p[["product", "category", "supplier", "underage", "overage", "cr", "mu", "sd", "order",
                                          "cost_newsvendor", "cost_median"]].to_dict("records"),
                          "total_newsvendor": c_nv.sum(), "total_median": c_med.sum()})

# %% [markdown]
# ## ✱ Stretch · Dates before Eid
#
# Double the goodwill part of the underage for the dates (`dates_ramadan`), as Imad would before Eid al-Adha.
# How far does the order move, and how much does the expected cost of the old order rise?

# %% [markdown]
# ## Checkpoint
#
# 1. A product's critical ratio is 0.27. Order above or below the median?
# 2. Why is the weekly spread not seven times the daily spread?
# 3. The forecaster's 80% bands held only 74% of the time on Day 2. What does that do to these orders?
#
# <details><summary>Answers</summary>
#
# 1. Below: a leftover costs more than a shortage, so the 0.27 quantile.
# 2. Independent days add variances, not spreads: the weekly sd is the square root of the sum of squares.
# 3. Bands too narrow make high quantiles too low: dry goods run short more often than planned.
#
# </details>
#
# **Carry forward.** The order table is in `data/reference/d3_1.json`; Lab 3.2 starts from the supplied
# targets (`U.week_inputs()`).
