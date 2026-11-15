"""Helpers shared by every lab of DS-404 Advanced Modelling for Structured Data.

The case: Bustan Food Wholesale, a fictional food wholesaler (depots in Riyadh, Jeddah, Dammam and Buraidah,
and a new northern region around Hail and Tabuk). All data is synthetic (prepare_data.py, seed 404).

    import bustan as U
    o = U.orders()                         # 80,000 orders: features, delivery hours, late
    X, y = U.xy(o)                         # the model features and the late flag
    Y = U.demand()                         # 400 daily series, one column each
    q, info = U.zero_shot(contexts, ...)   # a pretrained forecaster in Colab, the stand-in elsewhere
    U.check(cond, "what is true")          # a tick, or stops with what went wrong

Quick mode: BUSTAN_QUICK=1 makes every lab smaller (fewer rows, series, repeats, configurations).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"
QUICK = os.getenv("BUSTAN_QUICK", "0") == "1"
WRITE_REF = os.getenv("BUSTAN_WRITE_REF", "0") == "1"
IN_COLAB = "google.colab" in sys.modules
SEED = 404

H = 28                                          # the forecast horizon, in days
QL = (0.1, 0.5, 0.9)                            # the quantiles every forecaster returns
ORIGINS = pd.to_datetime(["2025-10-05", "2025-11-16", "2026-01-04", "2026-02-15", "2026-03-15", "2026-05-10"])
SCORED = ORIGINS[2:]                            # the four origins Labs 2.1 to 2.3 score
STATES = ["none", "pre_ramadan", "ramadan", "pre_eid", "eid"]
MODELS = ["seasonal_naive", "smoothing", "lightgbm"]   # the supplied benchmarks, in benchmarks.npz order
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)          # empty slices: products not launched yet
warnings.filterwarnings("ignore", message=".*non-interactive.*")    # scripts draw figures without a screen


def pick(full, quick):
    """The small value in quick mode (BUSTAN_QUICK=1), the full one otherwise."""
    return quick if QUICK else full


def check(cond, msg: str) -> None:
    if not cond:
        raise AssertionError("✗ " + msg)
    print("✓", msg)


def style():
    import matplotlib
    if not IN_COLAB and "ipykernel" not in sys.modules:
        matplotlib.use("Agg")                    # scripts and the notebook runner draw without a screen
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    return plt


# ------------------------------------------------------------------ files
_cache: dict = {}


def calendar() -> pd.DataFrame:
    """One row per day, indexed by date: dow, hijri_year, state and one flag per holiday state."""
    if "cal" not in _cache:
        _cache["cal"] = pd.read_csv(DATA / "calendar.csv", parse_dates=["date"]).set_index("date")
    return _cache["cal"].copy()


def orders(regions=None) -> pd.DataFrame:
    if "orders" not in _cache:
        _cache["orders"] = pd.read_csv(DATA / "orders.csv.gz", parse_dates=["date"])
    o = _cache["orders"]
    if regions is not None:
        o = o[o.region.isin([regions] if isinstance(regions, str) else regions)]
    return o.reset_index(drop=True).copy()


FEATURES = ["distance_km", "lines", "cases", "weight_kg", "chilled_share", "promised_hours", "truck_util",
            "driver_tenure_m", "ramadan", "pre_eid", "dow", "segment"]
SEGMENTS = ["hypermarket", "grocery", "horeca", "institutional"]


def xy(o: pd.DataFrame, target: str = "late"):
    """Model features (numbers only: the segment as a code) and the target."""
    X = o.assign(dow=o.date.dt.day_of_week, segment=o.segment.map({s: i for i, s in enumerate(SEGMENTS)}))[FEATURES]
    return X.astype(float), o[target].to_numpy()


def demand() -> pd.DataFrame:
    """Wide table: one row per day, one column per series (cases); NaN before a product's launch."""
    if "demand" not in _cache:
        z = np.load(DATA / "demand.npz")
        idx = pd.date_range(str(z["start"][0]), periods=z["y"].shape[1], freq="D")
        _cache["demand"] = pd.DataFrame(z["y"].T.astype(float), index=idx, columns=[str(s) for s in z["series"]])
    return _cache["demand"].copy()


def meta() -> pd.DataFrame:
    return pd.read_csv(DATA / "series_meta.csv", parse_dates=["launch"])


def products() -> pd.DataFrame:
    return pd.read_csv(DATA / "products.csv")


def js(path: str):
    return json.loads((DATA / path).read_text(encoding="utf-8"))


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def reference(name: str):
    """A supplied reference result (data/reference/<name>.json): what the safety nets fall back on."""
    return json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))


def save_reference(name: str, obj) -> None:
    """Write a reference result; only the course's own run does this (BUSTAN_WRITE_REF=1)."""
    if WRITE_REF and not QUICK:
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(_plain(obj), indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ tables: folds, baseline, the pretrained model
def grouped_folds(groups, k: int = 5, seed: int = SEED):
    """k folds that keep every group (a customer) on one side; groups are shuffled with the seed."""
    g = pd.Series(groups).astype(str).to_numpy()
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % k))
    f = np.array([fold_of[x] for x in g])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def random_folds(n: int, k: int = 5, seed: int = SEED):
    rng = np.random.default_rng(seed)
    f = rng.permutation(np.arange(n) % k)
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def lgbm(**params):
    import lightgbm as lgb
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4, random_state=SEED)
    base.update(params)
    return lgb.LGBMClassifier(**base)


def lgbm_reg(**params):
    import lightgbm as lgb
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=SEED)
    base.update(params)
    return lgb.LGBMRegressor(**base)


def sample_config(rng) -> dict:
    """One configuration of the random search (Lab 1.1's search space)."""
    return {"learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
            "num_leaves": int(rng.choice([7, 15, 31, 63])),
            "min_child_samples": int(rng.choice([10, 20, 50, 100])),
            "n_estimators": int(rng.choice([100, 200, 400])),
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "reg_lambda": float(rng.choice([0.0, 1.0, 5.0]))}


class StandInTabular:
    """THE STAND-IN, not a pretrained model. It takes the place of a tabular foundation model where none can run.

    Like an in-context learner it does no tuning and no search: it reads the training rows and predicts. Inside it
    averages a logistic regression on spline features with a nearest-neighbour vote. It answers in seconds on a laptop.
    """

    name = "stand-in (spline logistic + nearest neighbours; not a pretrained model)"

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X, y):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
        X = pd.DataFrame(X).reset_index(drop=True)
        num = [c for c in X.columns if c != "segment"]
        cat = [c for c in X.columns if c == "segment"]
        ct = ColumnTransformer([("s", make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3)), num),
                                ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
        self.lr = make_pipeline(ct, LogisticRegression(C=0.3, max_iter=2000)).fit(X, y)
        ct2 = ColumnTransformer([("s", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
        k = int(min(self.k, max(5, len(X) // 10)))
        self.knn = make_pipeline(ct2, KNeighborsClassifier(n_neighbors=k, weights="distance")).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        p = 0.6 * self.lr.predict_proba(X)[:, 1] + 0.4 * self.knn.predict_proba(X)[:, 1]
        return np.column_stack([1 - p, p])


def tabular_model():
    """The pretrained tabular model (TabICL) in a Colab runtime that has it, the stand-in everywhere else.

    Returns (model, info). info["live"] is True only when a real pretrained model answers.
    """
    if os.getenv("BUSTAN_PRETRAINED", "auto") != "standin":
        try:
            from tabicl import TabICLClassifier       # installed only in Colab: its weights download there
            return TabICLClassifier(n_estimators=pick(8, 4)), {"model": "TabICL (pretrained)", "live": True}
        except Exception:
            pass
    return StandInTabular(), {"model": StandInTabular.name, "live": False}


OLD_REGIONS = ["central", "western", "eastern"]


def curve_split(test_share: float = 0.2):
    """Lab 1.3's split of the established regions: a pool to draw training sizes from, and a fixed test set of
    held-out customers (no customer is on both sides)."""
    o = orders(OLD_REGIONS)
    cust = np.sort(o.customer.unique())
    rng = np.random.default_rng(SEED)
    test_c = set(rng.choice(cust, int(len(cust) * test_share), replace=False))
    is_test = o.customer.isin(test_c)
    return o[~is_test].reset_index(drop=True), o[is_test].reset_index(drop=True)


def cv_auc(model_fn, X, y, folds) -> float:
    s = []
    for tr, te in folds:
        m = model_fn().fit(X.iloc[tr], y[tr])
        s.append(auc(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(s))


def tuned_ensemble(X, y, groups, Xtest, budget: int = 8, k: int = 3, top: int = 3, seed: int = SEED):
    """Lab 1.1's procedure as one call: `budget` random configurations scored by grouped k-fold AUC, the best
    `top` refitted on all rows, their probabilities averaged on Xtest."""
    rng = np.random.default_rng(seed)
    folds = grouped_folds(groups, k, seed)
    cfgs = [sample_config(rng) for _ in range(budget)]
    cv = [cv_auc(lambda c=c: lgbm(**c), X, y, folds) for c in cfgs]
    best = np.argsort(cv)[::-1][:top]
    p = np.mean([lgbm(**cfgs[i]).fit(X, y).predict_proba(Xtest)[:, 1] for i in best], axis=0)
    return p, {"fits": budget * k + top, "cv_auc": [cv[i] for i in best], "configs": [cfgs[i] for i in best]}


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def logloss(y, p) -> float:
    from sklearn.metrics import log_loss
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def mean_ci(x, level: float = 0.95):
    """Mean and a t interval of independent repeats."""
    from scipy import stats
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, m, m
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def bootstrap_ci(d, n: int = 2000, level: float = 0.95, seed: int = SEED):
    """Mean of paired differences and a percentile bootstrap interval."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    b = rng.choice(d, (n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.quantile(b, (1 - level) / 2)), float(np.quantile(b, (1 + level) / 2))


# ------------------------------------------------------------------ series: states, scales, scores
def state_codes(index) -> np.ndarray:
    cal = calendar()
    return cal.state.reindex(index).map({s: i for i, s in enumerate(STATES)}).to_numpy()


def scale_of(y: np.ndarray) -> float:
    """MASE scale: mean absolute weekly-naive error over the last 365 days of history (at least 14 days)."""
    y = y[~np.isnan(y)][-365:]
    if len(y) < 14:
        return np.nan
    return max(float(np.mean(np.abs(y[7:] - y[:-7]))), 0.1)


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def scores(y: np.ndarray, q: np.ndarray, scale: np.ndarray) -> dict:
    """Scores of forecasts q [series, 3 quantiles, h] against actuals y [series, h], each series by its scale.

    Returns arrays per series: mase (median against y), sql (scaled quantile loss, mean over the three
    quantiles times two, so that the median term alone equals MASE) and cover80 (share of days inside q0.1..q0.9).
    """
    s = scale[:, None]
    mase = np.nanmean(np.abs(y - q[:, 1]) / s, axis=1)
    sql = 2 * np.nanmean(np.stack([pinball(y, q[:, j], t) for j, t in enumerate(QL)]), axis=(0, 2)) / scale
    cover = np.nanmean(np.where(np.isnan(y), np.nan, (y >= q[:, 0]) & (y <= q[:, 2])), axis=1)
    return {"mase": mase, "sql": sql, "cover80": cover}


def contexts_at(Y: pd.DataFrame, origin, length: int = 512):
    """For each series, its history up to and including the origin (the last `length` days, NaN dropped)."""
    hist = Y.loc[:origin].iloc[-length:]
    return [hist[c].dropna().to_numpy() for c in Y.columns], hist.index


def actuals(Y: pd.DataFrame, origin) -> np.ndarray:
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=H, freq="D")
    return Y.reindex(days).to_numpy().T


def series_subset() -> np.ndarray:
    """Indices of the series a lab uses: all 400, or every fourth in quick mode (100, every group kept)."""
    return np.arange(0, 400, pick(1, 4))


def score_long(Y: pd.DataFrame, fc: dict, origins, days=None) -> pd.DataFrame:
    """One row per model, origin and series: mase, sql, cover80.

    fc maps a model name to forecasts [origin, series, 3, h] at `origins`, series in Y's column order.
    days: optional function (origin -> boolean mask of the h days) to score a subset of days, such as holidays.
    """
    m = meta().set_index("series")
    rows = []
    for i, o in enumerate(origins):
        y = actuals(Y, o)
        sc = np.array([scale_of(Y[c].loc[:o].to_numpy()) for c in Y.columns])
        mask = days(o) if days is not None else np.ones(H, bool)
        if not mask.any():
            continue
        for name, q in fc.items():
            s = scores(y[:, mask], q[i][:, :, mask], sc)
            rows.append(pd.DataFrame({"model": name, "origin": o, "series": Y.columns, "group": m.group.reindex(Y.columns).to_numpy(),
                                      **{k: v for k, v in s.items()}}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["mase"])


def benchmarks():
    """The supplied rolling-origin backtests: q [model, origin, series, quantile, day] for MODELS at ORIGINS."""
    z = np.load(REF / "benchmarks.npz")
    return z["q"].astype(float), [str(s) for s in z["series"]]


def zero_shot_supplied():
    """The supplied zero-shot forecasts (written by the stand-in): q [covariates 0/1, origin, series, quantile, day]."""
    z = np.load(REF / "zero_shot.npz")
    return z["q"].astype(float), str(z["answered_by"][0])


# ------------------------------------------------------------------ the zero-shot forecaster
def _standin_one(y: np.ndarray, past: np.ndarray | None, fut: np.ndarray | None, h: int, dow_next: int) -> np.ndarray:
    n = len(y)
    if n == 0:
        return np.zeros((3, h))                      # a product not launched yet
    m = np.ones(len(STATES))
    if past is not None and n >= 60:
        base = pd.Series(np.where(past == 0, y, np.nan)).rolling(56, min_periods=14).mean().shift(1).to_numpy()
        ok = ~np.isnan(base) & (base > 0)
        r = np.where(ok, y / np.where(ok, base, 1), np.nan)
        r_none = np.nanmean(r[(past == 0) & ok]) if np.any((past == 0) & ok) else np.nan
        for s in range(1, len(STATES)):
            sel = (past == s) & ok
            if sel.sum() >= 5 and r_none and r_none > 0:
                m[s] = float(np.clip(np.nanmean(r[sel]) / r_none, 0.1, 5.0))
        yd = y / m[past]
    else:
        yd = y.copy()
    k = min(56, n)
    last = yd[-k:]
    dows = (np.arange(n - k, n) - n + dow_next) % 7                 # weekday of each context day
    lvl = np.zeros(7)
    for d in range(7):
        v = last[dows == d]
        lvl[d] = np.median(v) if len(v) else (np.median(last) if len(last) else 0.0)
    e = last - lvl[dows]
    lo, hi = (np.quantile(e, [0.1, 0.9]) if len(e) >= 7 else (-np.std(last), np.std(last)))
    fd = (np.arange(h) + dow_next) % 7
    mult = m[fut] if fut is not None else np.ones(h)
    q50 = lvl[fd] * mult
    q10 = np.maximum((lvl[fd] + lo) * mult, 0)
    q90 = np.maximum((lvl[fd] + hi) * mult, q50)
    return np.stack([np.minimum(q10, q50), q50, q90])


def _chronos(contexts, index, h, past_states, future_states):
    import torch  # noqa: F401  (present in Colab)
    from chronos import Chronos2Pipeline
    if "chronos" not in _cache:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _cache["chronos"] = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
    pipe = _cache["chronos"]
    ctx, fut = [], []
    for i, y in enumerate(contexts):
        if len(y) == 0:
            y = np.zeros(1)                          # a product not launched yet: Chronos needs one value
        days = index[-len(y):]
        d = pd.DataFrame({"id": i, "timestamp": days, "target": y})
        f = pd.DataFrame({"id": i, "timestamp": pd.date_range(index[-1] + pd.Timedelta(days=1), periods=h, freq="D")})
        if past_states is not None:
            for s in range(1, len(STATES)):
                d[STATES[s]] = (past_states[-len(y):] == s).astype(float)
                f[STATES[s]] = (future_states == s).astype(float)
        ctx.append(d)
        fut.append(f)
    pred = pipe.predict_df(pd.concat(ctx), future_df=pd.concat(fut) if past_states is not None else None,
                           prediction_length=h, quantile_levels=list(QL), id_column="id",
                           timestamp_column="timestamp", target="target")
    pred = pred.sort_values(["id", "timestamp"])
    return np.stack([pred[str(t)].to_numpy().reshape(len(contexts), h) for t in QL], axis=1)


def zero_shot(contexts, index, h: int = H, covariates: bool = False):
    """Zero-shot quantile forecasts for each context: array [series, 3 quantiles, h], and who answered.

    contexts: list of 1-D arrays (each series' history, ending on the last day of `index`).
    covariates: when True, the Hijri holiday states of the context and of the next h days are given as
    known covariates (pre-Ramadan, Ramadan, pre-Eid, Eid).
    In a Colab runtime with chronos-forecasting installed, Chronos-2 answers; elsewhere the stand-in does.
    """
    t0 = time.perf_counter()
    past = state_codes(index) if covariates else None
    fut_days = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    fut = state_codes(fut_days) if covariates else None
    if os.getenv("BUSTAN_PRETRAINED", "auto") != "standin":
        try:
            q = _chronos(contexts, index, h, past, fut)
            return np.clip(q, 0, None), {"model": "Chronos-2 (pretrained)", "live": True, "seconds": time.perf_counter() - t0}
        except ImportError:
            pass
    dow_next = int(fut_days[0].day_of_week)
    q = np.stack([_standin_one(y, None if past is None else past[-len(y):], fut, h, dow_next) for y in contexts])
    return q, {"model": "stand-in (weekday profile with in-context holiday multipliers; not a pretrained model)",
               "live": False, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------ from forecasts to decisions
def weekly_quantiles(q: np.ndarray, taus) -> np.ndarray:
    """Quantiles of a 7-day total from daily q [items, 3, 7], assuming independent days and a normal shape.

    The spread of each day comes from its 10% to 90% range (2.563 standard deviations). Independence is an
    assumption Lab 3.4 tests.
    """
    from scipy import stats
    mu = q[:, 1].sum(axis=1)
    sd = np.sqrt((((q[:, 2] - q[:, 0]) / 2.563) ** 2).sum(axis=1))
    z = stats.norm.ppf(np.atleast_1d(taus))
    out = mu[:, None] + sd[:, None] * z[None, :] if np.ndim(taus) == 0 else mu[:, None] + sd[:, None] * z[None, :]
    return np.maximum(out, 0)


PLAN_ORIGIN = 5                                 # Day 3 plans the week after 2026-05-10 (11 to 17 May)


def daily_forecasts(model: str, origin_index: int = PLAN_ORIGIN) -> np.ndarray:
    """Supplied daily forecasts [20 Day 3 products, 3, 28] of one model: seasonal_naive, smoothing, lightgbm,
    zero_shot or zero_shot_cov."""
    p = day3_products()
    cols = list(demand().columns)
    idx = [cols.index(s) for s in p.series]
    if model.startswith("zero_shot"):
        Z, _ = zero_shot_supplied()
        return Z[1 if model.endswith("cov") else 0, origin_index][idx]
    B, _ = benchmarks()
    return B[MODELS.index(model), origin_index][idx]


def week_inputs(model: str = "zero_shot_cov", origin_index: int = PLAN_ORIGIN, week: int = 0) -> pd.DataFrame:
    """Day 3's planning table for one week: the 20 products with costs, critical ratio, the weekly forecast
    (mean and spread from the daily quantiles, independent days) and the newsvendor quantity in cases."""
    from scipy import stats
    p = day3_products()
    q = daily_forecasts(model, origin_index)[:, :, 7 * week:7 * week + 7]
    mu = q[:, 1].sum(axis=1)
    sd = np.sqrt((((q[:, 2] - q[:, 0]) / 2.563) ** 2).sum(axis=1))
    p["cr"] = p.underage / (p.underage + p.overage)
    p["mu"], p["sd"] = mu, sd
    p["target"] = np.ceil(np.maximum(mu + sd * stats.norm.ppf(p.cr), 0))
    return p


def day3_products() -> pd.DataFrame:
    """The 20 products Day 3 plans for, at the Riyadh depot, with their costs per case."""
    p = products()
    pick20 = (p[p.group.isin(["smooth", "seasonal"])].groupby("supplier", group_keys=False)
              .apply(lambda d: d.head(5)))
    pick20 = pick20.reset_index(drop=True)
    pick20["series"] = pick20["product"] + "-RUH"
    margin = pick20.price - pick20.unit_cost
    weekly_hold = pick20.unit_cost * 0.004 * 7                                  # cost of capital and space per week
    spoil = np.where(pick20.shelf_days <= 30, 0.6 * pick20.unit_cost, 0.0)       # a chilled leftover is mostly lost
    pick20["underage"] = (margin + 0.5 * margin).round(2)                        # lost margin and the customer's goodwill
    pick20["overage"] = (weekly_hold + spoil).round(2)
    return pick20
