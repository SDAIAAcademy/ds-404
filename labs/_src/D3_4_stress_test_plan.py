# %% [markdown]
# # Lab 3.4 · Stress-test the plan
#
# Day 3 · Session 4 (Uncertainty in the plan) · 25 min.
#
# The plan of Lab 3.2 is fixed: the cases ordered for 11 to 17 May 2026. Before Ahlam signs it, she wants the
# shortage risk, not only the expected cost. You simulate 2,000 weeks of demand from the forecast, run the plan
# through them, and report the shortage probability and fill rate per product. Then you see what the
# assumption of Lab 3.1 (independent days, bands taken at face value) hides.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** The plan in `data/reference/d3_2.json`, the daily forecasts it was made from
# (`U.daily_forecasts("zero_shot_cov")`, written by the stand-in) and the same forecaster's backtests at the five
# earlier origins (for the error structure).
#
# **Tasks**
# - [ ] 1 · Scenarios from the forecast (6')
# - [ ] 2 · Run the plan through them (6')
# - [ ] 3 · Shortage risk per product (6')
# - [ ] 4 · Correlated against independent (7')
# - [ ] ✱ Stretch · Re-solve the plan on 200 scenarios and compare the risk

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
plan = pd.DataFrame(U.reference("d3_2")["plan"])
wi = U.week_inputs()
Y = U.demand()
IDX = [list(Y.columns).index(s) for s in wi.series]
q = U.daily_forecasts("zero_shot_cov")[:, :, :7]           # [20 products, 3 quantiles, 7 days]
MU = q[:, 1]
SD = np.maximum((q[:, 2] - q[:, 0]) / 2.563, 0.5)
print(f"plan: {int(plan.order.sum()):,} cases for 20 products; forecast mean {MU.sum():,.0f} cases")

# %% [markdown]
# ## 1 · Scenarios from the forecast (6')
#
# Lab 3.1 assumed each day independent and each band right. Check both on the forecaster's own past errors:
# standardise the errors at the five earlier origins, `z = (actual - median) / sd`, and measure
#
# - `scale`: the standard deviation of `z` (1 if the bands were right);
# - `rho_prod`: the correlation of `z` between two products on the same day;
# - `rho_day`: the correlation of `z` between two days of the same product in the same week.
#
# Then draw scenarios with one shock shared by all products in a scenario week, one per product-week, and one per
# day, mixed to give those correlations.
#
# ✏️ **Exercise** · Complete `scenarios`: return weekly totals `[n, 20]`, whole cases, never below zero.

# %%
def past_errors():
    Z = []
    for oi in range(U.PLAN_ORIGIN):
        f = U.daily_forecasts("zero_shot_cov", oi)
        y = U.actuals(Y, U.ORIGINS[oi])[IDX]
        Z.append((y - f[:, 1]) / np.maximum((f[:, 2] - f[:, 0]) / 2.563, 0.5))
    return np.stack(Z)                                       # [origins, products, 28 days]


Z = past_errors()
W = Z.reshape(Z.shape[0], 20, 4, 7)
scale = float(np.nanstd(Z))
days = W.reshape(-1, 7)
days = days[~np.isnan(days).any(axis=1)]
c7 = np.corrcoef(days.T)
rho_day = float(c7[np.triu_indices(7, 1)].mean())
flat = np.ma.masked_invalid(Z.transpose(1, 0, 2).reshape(20, -1))
c20 = np.ma.corrcoef(flat)
rho_prod = float(c20[np.triu_indices(20, 1)].mean())
print(f"past errors: scale {scale:.2f} | rho_prod {rho_prod:.2f} | rho_day {rho_day:.2f}")


def scenarios(n: int, rho_prod: float, rho_day: float, scale: float, seed: int = U.SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.sqrt(rho_prod)                                    # shared by all products in a scenario week
    b = np.sqrt(max(rho_day - rho_prod, 0.0))                # shared by the days of one product
    c = np.sqrt(max(1 - a ** 2 - b ** 2, 0.0))               # the rest, day by day
    # BEGIN SOLUTION
    z = (a * rng.standard_normal((n, 1, 1)) + b * rng.standard_normal((n, 20, 1))
         + c * rng.standard_normal((n, 20, 7)))
    daily = np.maximum(MU[None] + scale * SD[None] * z, 0)
    return np.round(daily.sum(axis=2))
    # END SOLUTION

# %%
try:
    D_ind = scenarios(2000, 0.0, 0.0, 1.0)
except NotImplementedError:
    def scenarios(n, rho_prod, rho_day, scale, seed=U.SEED):
        rng = np.random.default_rng(seed)
        a, b = np.sqrt(rho_prod), np.sqrt(max(rho_day - rho_prod, 0.0)); c = np.sqrt(max(1 - a ** 2 - b ** 2, 0.0))
        z = a * rng.standard_normal((n, 1, 1)) + b * rng.standard_normal((n, 20, 1)) + c * rng.standard_normal((n, 20, 7))
        return np.round(np.maximum(MU[None] + scale * SD[None] * z, 0).sum(axis=2))
    D_ind = scenarios(2000, 0.0, 0.0, 1.0)
U.check(D_ind.shape == (2000, 20) and (D_ind >= 0).all(), "2,000 weeks of demand for 20 products")
print(f"mean simulated demand {D_ind.mean(axis=0).sum():,.0f} cases a week")

# %% [markdown]
# ## 2 · Run the plan through them (6')
#
# ✏️ **Exercise** · For each scenario and product, the cases short (`shortfall`) and the cases sold (`sold`).

# %%
def run_plan(order: np.ndarray, D: np.ndarray):
    # BEGIN SOLUTION
    shortfall = np.maximum(D - order[None], 0)
    sold = np.minimum(D, order[None])
    return shortfall, sold
    # END SOLUTION

# %%
order = plan.order.to_numpy(float)
try:
    short, sold = run_plan(order, D_ind)
except NotImplementedError:
    run_plan = lambda order, D: (np.maximum(D - order[None], 0), np.minimum(D, order[None]))
    short, sold = run_plan(order, D_ind)
U.check(np.allclose(short + sold, D_ind), "every case of demand is either sold or short")

# %% [markdown]
# ## 3 · Shortage risk per product (6')
#
# ✏️ **Exercise** · Per product: `p_short`, the share of scenarios with any shortfall, and `fill_rate`, cases
# sold over cases demanded. Put next to them `1 - cr`, the shortage probability the newsvendor aimed for.

# %%
def risk_table(short, sold, D) -> pd.DataFrame:
    # BEGIN SOLUTION
    return pd.DataFrame({"p_short": (short > 0).mean(axis=0), "fill_rate": sold.sum(axis=0) / np.maximum(D.sum(axis=0), 1)})
    # END SOLUTION

# %%
try:
    risk = risk_table(short, sold, D_ind)
except NotImplementedError:
    risk = pd.DataFrame({"p_short": (short > 0).mean(axis=0), "fill_rate": sold.sum(axis=0) / np.maximum(D_ind.sum(axis=0), 1)})
risk.insert(0, "product", plan["product"])
risk.insert(1, "category", plan.category)
risk["aimed"] = 1 - plan.cr
risk["cut"] = plan.cut
print(risk.round(3).to_string(index=False))
U.check(risk.p_short.between(0, 1).all(), "shortage probabilities are probabilities")

# %% [markdown]
# **Look at.** Products the programme did not cut run short about as often as `1 - cr` says. The product it cut
# hardest runs short nearly every week: Lab 3.2 put most of the cut on one product, to fit one truck.

# %% [markdown]
# ## 4 · Correlated against independent (7')
#
# Three sets of 2,000 weeks: **independent** (what the plan assumed), **as measured** (the scale and correlations
# from Task 1) and **stress** (a shared shock with correlation 0.3 across products and days, the kind of week a
# heatwave or an early Eid rush brings). For each: the mean shortage probability, the chance that three or
# more of the 14 products meant to be in stock (critical ratio above 0.5, not cut below half its target) run short in the same week,
# the fill rate, and the 95th percentile of the week's shortage cost.

# %%
def summary(D, name: str) -> dict:
    s, so = run_plan(order, D)
    cost = (s * plan_under[None]).sum(axis=1)
    return {"scenarios": name, "mean_p_short": (s > 0).mean(), "p_3_or_more_in_stock_short": ((s[:, IN_STOCK] > 0).sum(axis=1) >= 3).mean(),
            "fill_rate": so.sum() / D.sum(), "short_cost_p95": np.quantile(cost, 0.95)}


plan_under = wi.set_index("product").underage.reindex(plan["product"]).to_numpy()
IN_STOCK = ((plan.cr > 0.5) & (plan.cut < 0.5 * plan.target)).to_numpy()               # the products meant to be in stock
sets = {"independent": scenarios(2000, 0.0, 0.0, 1.0),
        "as measured": scenarios(2000, max(rho_prod, 0), max(rho_day, rho_prod, 0), scale),
        "stress": scenarios(2000, 0.3, 0.3, scale)}
comp = pd.DataFrame([summary(D, k) for k, D in sets.items()])
print(comp.round(3).to_string(index=False))
U.check(IN_STOCK.sum() == 14, "14 products are meant to be in stock")
U.check(comp.p_3_or_more_in_stock_short.iloc[2] > comp.p_3_or_more_in_stock_short.iloc[0], "shared shocks make several products run short together")
U.check(comp.short_cost_p95.iloc[1] > comp.short_cost_p95.iloc[0], "the measured errors fatten the tail of the shortage cost")
U.save_reference("d3_4", {"scale": scale, "rho_prod": rho_prod, "rho_day": rho_day,
                          "risk": risk.to_dict("records"), "comparison": comp.to_dict("records")})

fig, ax = plt.subplots(figsize=(7, 3.2))
for k, D in sets.items():
    s, _ = run_plan(order, D)
    ax.hist((s[:, IN_STOCK] > 0).sum(axis=1), bins=np.arange(0, 15) - 0.5, histtype="step", lw=2, label=k, density=True)
ax.set_xlabel("products meant to be in stock that ran short in the week")
ax.set_ylabel("share of scenarios")
ax.legend()
plt.show()

# %% [markdown]
# **Look at.** The correlations in Bustan's past errors are small (0.09 across products, 0.04 across days);
# the larger gap is the scale: the bands were too narrow (Day 2 measured 74% coverage for 80% bands), so the
# chance that three or more in-stock products run short together nearly doubles. A shared shock adds little to the average
# risk but fattens the tail: the bad week's shortage cost (95th percentile) grows again. Independence with
# bands taken at face value is the optimistic corner of all three.

# %% [markdown]
# ## ✱ Stretch · Sample average approximation
#
# Write the programme of Lab 3.2 with the exact expected cost: for 200 of the "as measured" scenarios, add
# shortfall and leftover variables per scenario and product, minimise the average cost plus trucks, and keep
# the same capacity constraints. Compare the new plan's shortage risk with the old one. Does it still put the
# whole cut on one product?

# %% [markdown]
# ## Checkpoint
#
# 1. Two products each run short 10% of weeks. When is the chance both run short the same week far above 1%?
# 2. Why does a scale above 1 matter more for dry goods ordered at the 0.9 quantile than for dairy?
# 3. What do you add to the plan before Ahlam signs it?
#
# <details><summary>Answers</summary>
#
# 1. When their demand shares a shock (the same week, the same weather, the same holiday): independence
#    multiplies the probabilities; correlation does not.
# 2. A high quantile sits far in the tail, where a narrow band moves it most; a low quantile sits near the middle.
# 3. The shortage probability and fill rate per product under the measured error scale, and the chance that
#    several products run short together.
#
# </details>
