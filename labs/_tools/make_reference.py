"""Write the supplied states of the labs that need a long run (run after prepare_data.py).

    uv run python _tools/make_reference.py            # everything
    uv run python _tools/make_reference.py forecasts  # only the benchmark and zero-shot forecasts
    uv run python _tools/make_reference.py curve      # only the learning-curve scores at 50,000 orders

data/reference/benchmarks.npz   rolling-origin backtests of three benchmarks: a weekly seasonal naive, a
                                smoothing model (exponential smoothing on a weekday-adjusted series) and a tuned
                                global LightGBM with quantile objectives; six origins, 28 days, quantiles 0.1/0.5/0.9
data/reference/zero_shot.npz    zero-shot forecasts with and without holiday covariates at the same origins,
                                written by the STAND-IN (bustan.zero_shot outside Colab); rerun this script in a
                                Colab runtime with chronos-forecasting installed to write Chronos-2's instead
data/reference/curve_supplied.json  learning-curve scores at 50,000 orders (Lab 1.3), five repeats, both models
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import bustan as U  # noqa: E402

REF = LABS / "data" / "reference"
warnings.filterwarnings("ignore", category=RuntimeWarning)
REF.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ statistical benchmarks, vectorised over series
def _naive(Y, o):
    """Weekly seasonal naive at day index o: the last seven days repeated."""
    last = Y[:, o - 6:o + 1]
    return np.concatenate([last] * 4, axis=1)[:, :U.H]


def _smoothing(Y, o, dows):
    """Exponential smoothing of a weekday-adjusted series, alpha chosen per series by one-step error."""
    w = Y[:, o - 83:o + 1]
    wd = dows[o - 83:o + 1]
    m = np.nanmean(w, axis=1, keepdims=True)
    prof = np.ones((Y.shape[0], 7))
    for d in range(7):
        v = np.nanmean(w[:, wd == d], axis=1)
        prof[:, d] = np.where((m[:, 0] > 0) & ~np.isnan(v), v / np.where(m[:, 0] > 0, m[:, 0], 1), 1)
    prof = np.where(prof <= 0.05, 0.05, prof)
    z = Y[:, o - 364:o + 1] / prof[:, dows[o - 364:o + 1]]
    best_sse = np.full(Y.shape[0], np.inf)
    best_lvl = np.zeros(Y.shape[0])
    for a in (0.03, 0.08, 0.15, 0.3):
        lvl = np.full(Y.shape[0], np.nan)
        sse = np.zeros(Y.shape[0])
        for t in range(z.shape[1]):
            x = z[:, t]
            ok = ~np.isnan(x)
            new = np.isnan(lvl) & ok
            lvl[new] = x[new]
            err = np.where(ok & ~new, x - lvl, 0.0)
            sse += err ** 2
            lvl = np.where(ok & ~new, lvl + a * err, lvl)
        better = sse < best_sse
        best_sse[better], best_lvl[better] = sse[better], lvl[better]
    fd = dows[o + 1:o + 1 + U.H] if o + 1 + U.H <= len(dows) else (np.arange(U.H) + dows[o] + 1) % 7
    return np.nan_to_num(best_lvl)[:, None] * prof[:, fd]


def _with_quantiles(f, Y, o, dows):
    """Point forecasts at o, and 10% and 90% quantiles from the errors of eight earlier origins (in history only)."""
    q50 = f(Y, o, dows) if f is _smoothing else f(Y, o)
    errs = []
    for k in range(1, 9):
        p = o - 7 * k
        fc = f(Y, p, dows) if f is _smoothing else f(Y, p)
        n = min(U.H, o - p)
        errs.append(Y[:, p + 1:p + 1 + n] - fc[:, :n])
    e = np.concatenate(errs, axis=1)
    lo, hi = np.nanquantile(e, 0.1, axis=1), np.nanquantile(e, 0.9, axis=1)
    lo, hi = np.nan_to_num(lo), np.nan_to_num(hi)
    q10 = np.maximum(q50 + lo[:, None], 0)
    q90 = np.maximum(q50 + hi[:, None], q50)
    return np.stack([np.minimum(q10, q50), np.maximum(q50, 0), q90], axis=1)


# ------------------------------------------------------------------ the learned benchmark: one global LightGBM
def _frame(Y, o, states, dows, group_code, train: bool):
    """Rows for origin o: features known at o, for each series and each day ahead."""
    n = Y.shape[0]
    hist = Y[:, max(0, o - 364):o + 1]
    m7 = np.nanmean(hist[:, -7:], axis=1)
    m28 = np.nanmean(hist[:, -28:], axis=1)
    zero90 = np.nanmean(hist[:, -90:] == 0, axis=1)
    scale = np.nan_to_num(m28) + 1.0
    wdm = np.full((n, 7), np.nan)
    for d in range(7):
        idx = [i for i in range(o - 27, o + 1) if dows[i] == d]
        wdm[:, d] = np.nanmean(Y[:, idx], axis=1)
    age = (~np.isnan(Y[:, :o + 1])).sum(axis=1)
    rows = []
    for h in range(1, U.H + 1):
        t = o + h
        if t >= Y.shape[1]:
            break
        d = dows[t]
        f = {"h": np.full(n, h), "dow": np.full(n, d), "state": np.full(n, states[t]),
             "state_lag354": np.full(n, states[t - 354]), "lag354": Y[:, t - 354] / scale, "lag364": Y[:, t - 364] / scale,
             "wd4": wdm[:, d] / scale, "m7": m7 / scale, "m28_log": np.log1p(np.nan_to_num(m28)), "zero90": zero90,
             "group": group_code, "age": np.minimum(age, 400), "series": np.arange(n), "scale": scale,
             "y": Y[:, t] / scale}
        rows.append(pd.DataFrame(f))
    df = pd.concat(rows, ignore_index=True)
    if train:
        df = df[~np.isnan(df.y) & (df.age >= 28)]
    return df


FEATS = ["h", "dow", "state", "state_lag354", "lag354", "lag364", "wd4", "m7", "m28_log", "zero90", "group", "age"]


def _lgbm_backtest(Y, o, states, dows, group_code, params):
    import lightgbm as lgb
    train = pd.concat([_frame(Y, p, states, dows, group_code, True) for p in range(400, o - U.H + 1, 14)], ignore_index=True)
    test = _frame(Y, o, states, dows, group_code, False)
    out = np.zeros((Y.shape[0], 3, U.H))
    for j, tau in enumerate(U.QL):
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, verbose=-1, n_jobs=8, random_state=U.SEED, **params)
        m.fit(train[FEATS], train.y, categorical_feature=["dow", "state", "state_lag354", "group"])
        p = m.predict(test[FEATS]) * test.scale.to_numpy()
        out[:, j, :] = np.maximum(p.reshape(U.H, Y.shape[0]).T, 0)
    out[:, 0] = np.minimum(out[:, 0], out[:, 1])
    out[:, 2] = np.maximum(out[:, 2], out[:, 1])
    return out


def _tune_lgbm(Y, states, dows, group_code):
    """A small search on the last 56 days before the first origin: pick the configuration with the lowest median loss."""
    import lightgbm as lgb
    o0 = U.Y_INDEX.get_loc(U.ORIGINS[0])
    val_o = [o0 - 56, o0 - 28]
    train = pd.concat([_frame(Y, p, states, dows, group_code, True) for p in range(400, val_o[0] - U.H + 1, 14)], ignore_index=True)
    val = pd.concat([_frame(Y, p, states, dows, group_code, True) for p in val_o], ignore_index=True)
    best, best_loss = None, np.inf
    for cfg in ({"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50},
                {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 100},
                {"n_estimators": 300, "learning_rate": 0.08, "num_leaves": 15, "min_child_samples": 20},
                {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 200}):
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, verbose=-1, n_jobs=8, random_state=U.SEED, **cfg)
        m.fit(train[FEATS], train.y, categorical_feature=["dow", "state", "state_lag354", "group"])
        loss = float(np.mean(np.abs(m.predict(val[FEATS]) - val.y) * val.scale))
        print("  lightgbm config", cfg, "validation MAE", round(loss, 3))
        if loss < best_loss:
            best, best_loss = cfg, loss
    return best


def forecasts():
    Yd = U.demand()
    U.Y_INDEX = Yd.index
    Y = Yd.to_numpy().T
    dows = Yd.index.day_of_week.to_numpy()
    states = U.state_codes(Yd.index)
    meta = U.meta()
    group_code = meta.group.map({"smooth": 0, "seasonal": 1, "intermittent": 2, "short": 3}).to_numpy()
    t0 = time.time()
    params = _tune_lgbm(Y, states, dows, group_code)
    print("lightgbm chosen:", params)
    q = np.zeros((3, len(U.ORIGINS), Y.shape[0], 3, U.H), dtype="float32")
    for i, origin in enumerate(U.ORIGINS):
        o = Yd.index.get_loc(origin)
        q[0, i] = _with_quantiles(_naive, Y, o, dows)
        q[1, i] = _with_quantiles(_smoothing, Y, o, dows)
        q[2, i] = _lgbm_backtest(Y, o, states, dows, group_code, params)
        print(f"origin {origin.date()} done, {time.time() - t0:.0f} s")
    np.savez_compressed(REF / "benchmarks.npz", q=np.round(q, 3), series=np.array(Yd.columns, dtype=str),
                        models=np.array(U.MODELS), origins=np.array([o.strftime("%Y-%m-%d") for o in U.ORIGINS]))
    zs = np.zeros((2, len(U.ORIGINS), Y.shape[0], 3, U.H), dtype="float32")
    who = ""
    for i, origin in enumerate(U.ORIGINS):
        ctx, idx = U.contexts_at(Yd, origin)
        for c in (0, 1):
            fq, info = U.zero_shot(ctx, idx, covariates=bool(c))
            zs[c, i] = fq
            who = info["model"]
    np.savez_compressed(REF / "zero_shot.npz", q=np.round(zs, 3), answered_by=np.array([who]),
                        origins=np.array([o.strftime("%Y-%m-%d") for o in U.ORIGINS]))
    print("wrote benchmarks.npz and zero_shot.npz; zero-shot answered by:", who)


# ------------------------------------------------------------------ Lab 1.3: the largest training size
def curve():
    import lightgbm  # noqa: F401
    pool, test = U.curve_split()
    Xt, yt = U.xy(test)
    out = {"size": 50000, "repeats": [], "note": "written by _tools/make_reference.py; the pretrained column is the STAND-IN"}
    for r in range(5):
        t0 = time.perf_counter()
        tr = pool.sample(50000, random_state=U.SEED + r)
        X, y = U.xy(tr)
        p_gbm, info_gbm = U.tuned_ensemble(X, y, tr.customer, Xt, budget=8, seed=U.SEED + r)
        t1 = time.perf_counter()
        m, info = U.tabular_model()
        m.fit(X, y)
        p_pre = m.predict_proba(Xt)[:, 1]
        t2 = time.perf_counter()
        out["repeats"].append({"repeat": r, "gbm_auc": U.auc(yt, p_gbm), "pre_auc": U.auc(yt, p_pre),
                               "gbm_seconds": t1 - t0, "pre_seconds": t2 - t1, "pretrained": info["model"]})
        print(out["repeats"][-1])
    import json
    (REF / "curve_supplied.json").write_text(json.dumps(U._plain(out), indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    what = sys.argv[1:] or ["forecasts", "curve"]
    if "forecasts" in what:
        forecasts()
    if "curve" in what:
        curve()
