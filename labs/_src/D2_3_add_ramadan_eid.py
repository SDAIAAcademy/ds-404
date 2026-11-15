# %% [markdown]
# # Lab 2.3 · Ramadan and Eid as covariates
#
# Day 2 · Session 3 (Holidays that move) · 25 min.
#
# Ramadan 1447 started about eleven days earlier in the Gregorian calendar than Ramadan 1446. Demand has two
# cycles at once: the week (Friday is quiet) and the Hijri year. A lag of 364 days keeps the weekday and misses
# the Hijri date; a lag of 354 days hits the Hijri date and lands on the wrong weekday; 357 days (51 weeks)
# keeps the weekday and lands within three days of the Hijri date. A zero-shot forecaster sees none of this
# unless you give it the calendar as known future covariates. You measure the three lags, build the
# covariates, run the forecaster with them and split the error into holiday days and ordinary days.
#
# **Where to run it.** Chronos-2 in Colab (it accepts future covariates); the stand-in elsewhere, which
# estimates one multiplier per holiday state from the context it is given.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/demand.npz`, `data/calendar.csv`, `data/reference/benchmarks.npz`,
# `data/reference/zero_shot.npz`.
#
# **Tasks**
# - [ ] 1 · Three lags: 364, 357 and 354 days (5')
# - [ ] 2 · Build the future covariates (6')
# - [ ] 3 · Zero-shot with covariates (6')
# - [ ] 4 · Holiday weeks against ordinary weeks (8')
# - [ ] ✱ Stretch · Drop the pre-Eid flag and see which weeks suffer
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

S = U.series_subset()
Y = U.demand().iloc[:, S]
meta = U.meta().iloc[S].reset_index(drop=True)
cal = U.calendar()
B, _ = U.benchmarks()
B = B[:, :, S]
scored_i = [list(U.ORIGINS).index(o) for o in U.SCORED]


# %% [markdown]
# ## 1 · Three lags: 364, 357 and 354 days (5')
#
# Forecast each day from pre-Ramadan to the end of Eid al-Fitr 1447 by the value one lag earlier, for the
# Ramadan items.
#
# ✏️ **Exercise** · Return the mean absolute error, divided by each series' mean over the window, of the lag
# forecast over the given days, averaged over the given series.

# %%
def lag_error(Y: pd.DataFrame, lag: int, days: pd.DatetimeIndex) -> float:
    # BEGIN SOLUTION
    actual = Y.reindex(days)
    lagged = Y.reindex(days - pd.Timedelta(days=lag)).set_axis(days)
    err = (actual - lagged).abs().mean() / actual.mean()
    return float(err.replace([np.inf], np.nan).mean())
    # END SOLUTION

# %%
window = pd.date_range("2026-02-04", "2026-03-23", freq="D")      # pre-Ramadan to the last Eid al-Fitr day
seasonal = Y.loc[:, meta.category.str.endswith("ramadan").to_numpy()]
try:
    err = {lag: lag_error(seasonal, lag, window) for lag in (364, 357, 354)}
except NotImplementedError:
    def _le(Y, lag, days):
        a = Y.reindex(days); b = Y.reindex(days - pd.Timedelta(days=lag)).set_axis(days)
        return float(((a - b).abs().mean() / a.mean()).replace([np.inf], np.nan).mean())
    err = {lag: _le(seasonal, lag, window) for lag in (364, 357, 354)}
print(f"window {window[0].date()} to {window[-1].date()}; Ramadan items: {seasonal.shape[1]} series")
for lag, e in err.items():
    print(f"  lag {lag} days ({lag / 7:.1f} weeks): relative error {e:.3f}")
U.check(err[357] < min(err[364], err[354]), "the lag that keeps both the weekday and the Hijri date wins")
e364, e357, e354 = err[364], err[357], err[354]

# %% [markdown]
# ## 2 · Build the future covariates (6')
#
# The calendar holds one **state** per day: none, pre-Ramadan (14 days), Ramadan, pre-Eid (7 days) and Eid
# (four days). These are known in advance, so they are legitimate **future** covariates.
#
# ✏️ **Exercise** · For one origin, return the states of the next 28 days as a Series indexed by date.

# %%
def future_states(origin) -> pd.Series:
    # BEGIN SOLUTION
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=U.H, freq="D")
    return cal.state.reindex(days)
    # END SOLUTION

# %%
try:
    fs = {o: future_states(o) for o in U.SCORED}
except NotImplementedError:
    fs = {o: cal.state.reindex(pd.date_range(o + pd.Timedelta(days=1), periods=U.H, freq="D")) for o in U.SCORED}
for o, s in fs.items():
    print(str(o.date()), s.value_counts().to_dict())
U.check(all(len(s) == U.H for s in fs.values()), "28 future days per origin")
U.check(any((s == "ramadan").any() for s in fs.values()), "at least one scored window contains Ramadan")

# %% [markdown]
# ## 3 · Zero-shot with covariates (6')
#
# `U.zero_shot(contexts, index, covariates=True)` passes the states of the context and of the next 28 days.
# Run it at the four scored origins, without and with covariates.

# %%
out = {0: [], 1: []}
for o in U.SCORED:
    ctx, idx = U.contexts_at(Y, o)
    for c in (0, 1):
        q, info = U.zero_shot(ctx, idx, covariates=bool(c))
        out[c].append(q)
ZS, ZC = np.stack(out[0]), np.stack(out[1])
print("answered by:", info["model"])
if not info["live"]:
    print("!! STAND-IN: not a pretrained model. In Colab, Chronos-2 answers this cell.")
U.check(ZC.shape == ZS.shape, "forecasts with and without covariates")

# %% [markdown]
# ## 4 · Holiday weeks against ordinary weeks (8')
#
# ✏️ **Exercise** · Score the four methods on holiday days only and on ordinary days only. `U.score_long`
# takes a `days` function: origin to a boolean mask of the 28 days. Return a table with one row per model and
# the columns `holiday_mase`, `ordinary_mase`, `holiday_sql`, `ordinary_sql`.

# %%
fc = {"smoothing": B[1][scored_i], "LightGBM (tuned)": B[2][scored_i], "zero-shot": ZS, "zero-shot + covariates": ZC}


def holiday_split(Y, fc) -> pd.DataFrame:
    # BEGIN SOLUTION
    hol = U.score_long(Y, fc, U.SCORED, days=lambda o: (fs[o] != "none").to_numpy())
    ordi = U.score_long(Y, fc, U.SCORED, days=lambda o: (fs[o] == "none").to_numpy())
    a = hol.groupby("model")[["mase", "sql"]].mean().add_prefix("holiday_")
    b = ordi.groupby("model")[["mase", "sql"]].mean().add_prefix("ordinary_")
    return a.join(b)[["holiday_mase", "ordinary_mase", "holiday_sql", "ordinary_sql"]]
    # END SOLUTION

# %%
try:
    split = holiday_split(Y, fc)
except NotImplementedError:
    hol = U.score_long(Y, fc, U.SCORED, days=lambda o: (fs[o] != "none").to_numpy())
    ordi = U.score_long(Y, fc, U.SCORED, days=lambda o: (fs[o] == "none").to_numpy())
    split = hol.groupby("model")[["mase", "sql"]].mean().add_prefix("holiday_").join(
        ordi.groupby("model")[["mase", "sql"]].mean().add_prefix("ordinary_"))[["holiday_mase", "ordinary_mase", "holiday_sql", "ordinary_sql"]]
print(split.round(3).to_string())
gain = 1 - split.loc["zero-shot + covariates", "holiday_sql"] / split.loc["zero-shot", "holiday_sql"]
print(f"covariates cut the zero-shot holiday-day quantile loss by {gain:.0%}")
U.check(split.loc["zero-shot + covariates", "holiday_sql"] < split.loc["zero-shot", "holiday_sql"],
        "covariates lower the holiday-day loss")
hol_ramadan_items = U.score_long(Y, fc, U.SCORED, days=lambda o: (fs[o] != "none").to_numpy())
hol_ramadan_items = hol_ramadan_items[hol_ramadan_items.group == "seasonal"].groupby("model").sql.mean()
print("\nRamadan items, holiday days, quantile loss:\n", hol_ramadan_items.round(3).to_string())
U.save_reference("d2_3", {"answered_by": info["model"], "e364": e364, "e357": e357, "e354": e354, "split": split.reset_index().to_dict("records"),
                          "ramadan_items_holiday_sql": hol_ramadan_items.to_dict(), "gain": gain})

# %% [markdown]
# **Look at.** The tuned LightGBM had the holiday states as features from the start: the supplied backtest
# already knew the calendar. A fair comparison gives every method the same known-future information.

# %% [markdown]
# ## ✱ Stretch · Without the pre-Eid flag
#
# Copy `U.zero_shot` into a cell, merge `pre_eid` into `none` before the call, and score the holiday days
# again. Which origin's window loses most?

# %% [markdown]
# ## Checkpoint
#
# 1. Why does a 364-day lag miss Ramadan, and why does a 354-day lag not fix it alone?
# 2. Is it leakage to give the forecaster the Eid dates of the next 28 days?
# 3. Covariates help the zero-shot model on holiday days. What do you check on ordinary days?
#
# <details><summary>Answers</summary>
#
# 1. Ramadan follows the Hijri year of about 354 days, so it moves about eleven days earlier each Gregorian year;
#    354 days later is a different weekday, so the weekly pattern breaks. 357 days (51 weeks) keeps both close.
# 2. No: the calendar is published in advance. Leakage is anything not known at the origin, such as next week's sales.
# 3. That the ordinary-day error did not get worse: a covariate that adds noise elsewhere is not free.
#
# </details>
#
# **Carry forward.** The holiday split is in `data/reference/d2_3.json`; Day 3 plans from the zero-shot
# forecasts with covariates (supplied in `zero_shot.npz`).