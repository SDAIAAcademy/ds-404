"""Write the synthetic data of Bustan Food Wholesale (DS-404). Same bytes on every run (seed 404).

    uv run python prepare_data.py

Writes into data/:
    calendar.csv            one row per day: Hijri year, Ramadan, pre-Ramadan, Eid and pre-Eid flags
    orders.csv.gz           80,000 orders with delivery hours and the late flag (Days 1 and 4)
    demand.npz              400 daily demand series in cases (Days 2 and 3); NaN before a product's launch
    series_meta.csv         one row per series: product, depot, category, group, launch day
    products.csv            one row per product: supplier, costs, shelf life, case volume and weight
    trucks.json             truck and reefer capacity and cost, depot space (Day 3)
    leaderboard_claim.csv   a vendor benchmark table to check (Lab 1.4)
    adoption/               candidate cards, classification register, cost sheet (Lab 4.4)

The heavier supplied states (benchmark backtests, stand-in zero-shot forecasts, learning-curve scores at
50,000 orders) are written by _tools/make_reference.py, which reads these files.

Everything is synthetic. Hijri dates follow the Umm al-Qura calendar as published for 1445 to 1447.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
SEED = 404

START, END = pd.Timestamp("2023-07-19"), pd.Timestamp("2026-06-15")   # 1 Muharram 1445 to the end of 1447
HIJRI_YEARS = [(1445, "2023-07-19"), (1446, "2024-07-07"), (1447, "2025-06-26")]
RAMADAN = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-29"), ("2026-02-18", "2026-03-19")]
EID_FITR = ["2024-04-10", "2025-03-30", "2026-03-20"]
EID_ADHA = ["2024-06-16", "2025-06-06", "2026-05-27"]
STATES = ["none", "pre_ramadan", "ramadan", "pre_eid", "eid"]


def calendar() -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    cal = pd.DataFrame({"date": days})
    cal["dow"] = cal.date.dt.day_of_week                       # Monday 0 ... Friday 4, Saturday 5
    cal["hijri_year"] = 0
    for y, d in HIJRI_YEARS:
        cal.loc[cal.date >= d, "hijri_year"] = y
    state = pd.Series("none", index=cal.index)
    for a, b in RAMADAN:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        state[(cal.date >= a - pd.Timedelta(days=14)) & (cal.date < a)] = "pre_ramadan"
        state[(cal.date >= a) & (cal.date <= b)] = "ramadan"
    for e in EID_FITR + EID_ADHA:
        e = pd.Timestamp(e)
        pre = (cal.date >= e - pd.Timedelta(days=7)) & (cal.date < e) & (state != "ramadan")
        state[pre] = "pre_eid"
        state[(cal.date >= e) & (cal.date <= e + pd.Timedelta(days=3))] = "eid"
    # the last week of Ramadan is also the week before Eid al-Fitr: it keeps "ramadan", the stronger state
    cal["state"] = state
    for s in STATES[1:]:
        cal[s] = (state == s).astype(int)
    return cal


# ------------------------------------------------------------------ products and demand
CATS = {  # category: (group, supplier, chilled, shelf_days, base level range)
    "rice_flour_sugar": ("smooth", "S1", 0, 365, (40, 160)),
    "oil_canned": ("smooth", "S1", 0, 540, (25, 90)),
    "beverages": ("smooth", "S2", 0, 270, (30, 120)),
    "dairy_chilled": ("smooth", "S3", 1, 21, (35, 110)),
    "dates_ramadan": ("seasonal", "S4", 0, 180, (6, 25)),
    "juices_ramadan": ("seasonal", "S2", 0, 240, (8, 30)),
    "specialty_imports": ("intermittent", "S1", 0, 365, (0.4, 2.0)),
    "catering_supplies": ("intermittent", "S4", 0, 720, (0.5, 2.5)),
    "new_lines": ("short", "S2", 0, 300, (6, 30)),
}
PLAN = [("rice_flour_sugar", 12), ("oil_canned", 10), ("beverages", 10), ("dairy_chilled", 8),
        ("dates_ramadan", 10), ("juices_ramadan", 10), ("specialty_imports", 13), ("catering_supplies", 12),
        ("new_lines", 15)]                                    # 100 products
DEPOTS = [("RUH", "Riyadh", "central", 1.0), ("JED", "Jeddah", "western", 0.85),
          ("DMM", "Dammam", "eastern", 0.6), ("BUR", "Buraidah", "central", 0.4)]


def products(rng) -> pd.DataFrame:
    rows, k = [], 1
    for cat, n in PLAN:
        group, sup, chilled, shelf, (lo, hi) = CATS[cat]
        for _ in range(n):
            cost = float(np.round(rng.uniform(25, 160), 1))
            margin = float(np.round(rng.uniform(0.08, 0.22) if group != "intermittent" else rng.uniform(0.2, 0.35), 3))
            rows.append({
                "product": f"P{k:03d}", "category": cat, "group": group, "supplier": sup, "chilled": chilled,
                "shelf_days": shelf, "unit_cost": cost, "price": round(cost * (1 + margin), 1),
                "case_m3": float(np.round(rng.uniform(0.02, 0.06), 3)), "case_kg": float(np.round(rng.uniform(6, 22), 1)),
                "base": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
            })
            k += 1
    return pd.DataFrame(rows)


def demand(rng, cal: pd.DataFrame, prod: pd.DataFrame):
    T = len(cal)
    t = np.arange(T)
    dow = cal.dow.to_numpy()
    week = np.array([1.05, 1.08, 1.06, 1.1, 0.35, 0.8, 1.0])           # Friday low, Saturday lower
    M = {  # multiplier per holiday state and group
        "smooth": {"pre_ramadan": 1.30, "ramadan": 1.10, "pre_eid": 1.35, "eid": 0.30},
        "seasonal": {"pre_ramadan": 2.6, "ramadan": 2.2, "pre_eid": 1.6, "eid": 0.35},
        "intermittent": {"pre_ramadan": 1.2, "ramadan": 0.9, "pre_eid": 1.3, "eid": 0.2},
        "short": {"pre_ramadan": 1.4, "ramadan": 1.2, "pre_eid": 1.4, "eid": 0.3},
    }
    state = cal.state.to_numpy()
    series, meta = [], []
    for _, p in prod.iterrows():
        for code, name, region, scale in DEPOTS:
            g = p.group
            mult = np.ones(T)
            for s, m in M[g].items():
                mult[state == s] = m * rng.uniform(0.85, 1.15)
            if p.category == "dairy_chilled":
                mult[state == "ramadan"] *= 0.85
            trend = 1 + rng.normal(0.0, 0.08) * t / 365
            annual = 1 + 0.08 * np.sin(2 * np.pi * (t / 365.25) + rng.uniform(0, 2 * np.pi))  # summer and school terms
            prof = week * rng.uniform(0.9, 1.1, 7)
            mu = p.base * scale * rng.uniform(0.8, 1.2) * prof[dow] * mult * trend * annual
            if g == "intermittent":
                occur = rng.random(T) < np.clip(mu, 0, None) / (np.clip(mu, 0, None) + rng.uniform(2.0, 4.0))
                size = rng.gamma(2.0, rng.uniform(1.5, 3.0), T)
                y = np.where(occur, np.ceil(size), 0.0)
            else:
                shape = rng.uniform(6, 14)
                lam = rng.gamma(shape, np.clip(mu, 1e-6, None) / shape)
                y = rng.poisson(lam).astype(float)
            launch = 0
            if g == "short":
                launch = int(T - rng.integers(90, 180))
                ramp = np.clip((t - launch) / 30, 0, 1)
                y = np.round(y * ramp)
                y[:launch] = np.nan
            series.append(y)
            meta.append({"series": f"{p['product']}-{code}", "product": p["product"], "depot": code, "depot_name": name,
                         "region": region, "category": p.category, "group": g, "supplier": p.supplier,
                         "launch": cal.date.iloc[launch].strftime("%Y-%m-%d")})
    return np.array(series), pd.DataFrame(meta)


# ------------------------------------------------------------------ orders
SEG = ["hypermarket", "grocery", "horeca", "institutional"]
PROMISED = {"hypermarket": 24, "grocery": 48, "horeca": 12, "institutional": 24}
LINES = {"hypermarket": 22, "grocery": 7, "horeca": 11, "institutional": 14}
SEG_EFF = {"hypermarket": -0.10, "grocery": 0.05, "horeca": 0.18, "institutional": -0.05}
NORTH_START = pd.Timestamp("2025-08-24")


def orders(rng, cal: pd.DataFrame) -> pd.DataFrame:
    days = cal.date.to_numpy()
    state = cal.set_index("date").state

    def customers(n, region_depots, north: bool, start_id: int):
        seg = rng.choice(SEG, n, p=[0.08, 0.62, 0.14, 0.16] if north else [0.15, 0.45, 0.28, 0.12])
        if north:
            dist = np.exp(rng.normal(np.log(45), 0.75, n))
        else:
            dist = np.exp(rng.normal(np.log(18), 0.6, n))
        return pd.DataFrame({
            "customer": [f"C{start_id + i:05d}" for i in range(n)], "segment": seg,
            "depot": rng.choice(region_depots, n), "distance_km": np.round(np.clip(dist, 2, 320), 1),
            "u": rng.normal(0, 0.5, n), "size": rng.lognormal(0, 0.5, n),
        })

    old = customers(1500, ["RUH", "BUR", "JED", "DMM"], False, 1)
    new = customers(250, ["HAS", "TUU"], True, 7001)
    region = {"RUH": "central", "BUR": "central", "JED": "western", "DMM": "eastern", "HAS": "northern", "TUU": "northern"}

    def draw(cust, n, d0, d1, north: bool):
        w = cust["size"].to_numpy()
        idx = rng.choice(len(cust), n, p=w / w.sum())
        c = cust.iloc[idx].reset_index(drop=True)
        span = (d1 - d0).days + 1
        dates = d0 + pd.to_timedelta(rng.integers(0, span, n), unit="D")
        dates = pd.DatetimeIndex(dates)
        fri = dates.day_of_week == 4
        dates = dates.where(~fri, dates + pd.Timedelta(days=1))   # no Friday deliveries
        dates = dates.where(dates <= END, dates - pd.Timedelta(days=2))
        st = state.reindex(dates).to_numpy()
        lines = 1 + rng.poisson([LINES[s] for s in c.segment])
        cases = np.round(lines * rng.gamma(3, 1.6, n)).astype(int) + 1
        chilled = np.round(rng.beta(1.2, 4, n), 2)
        util = np.round(rng.beta(5, 4, n) if north else rng.beta(6, 3, n), 3)
        tenure = np.round(rng.gamma(2, 4, n) if north else rng.gamma(3, 10, n)).astype(int) + 1
        promised = np.array([PROMISED[s] for s in c.segment])
        ram = (st == "ramadan").astype(int)
        pre = (st == "pre_eid").astype(int)
        dist = c.distance_km.to_numpy()
        kd = 0.52 if north else 0.38                          # planted concept shift: distance costs more in the north
        r = (-0.9 + kd * np.log1p(dist / 20) + 0.25 * chilled * np.log1p(dist / 10)
             + 1.6 * np.clip(util - 0.80, 0, None) / 0.2 + np.array([SEG_EFF[s] for s in c.segment])
             + 0.12 * np.log(lines / 8) + 0.18 * ram + 0.32 * pre - 0.28 * np.minimum(tenure, 36) / 36
             - 0.10 * (promised == 48) + 0.8 * ((c.segment == "horeca").to_numpy() & (ram == 1)) + 0.7 * ((util > 0.85) & (chilled > 0.3))
             + 0.6 * ((lines > 25) & (promised == 12)) + 0.6 * ((tenure < 6) & (dist > 40))
             + c.u.to_numpy() + rng.normal(0, 0.26, n))
        hours = np.round(promised * np.exp(r), 1)
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"), "customer": c.customer, "depot": c.depot,
            "region": [region[d] for d in c.depot], "segment": c.segment, "distance_km": dist, "lines": lines,
            "cases": cases, "weight_kg": np.round(cases * rng.uniform(8, 16, n)), "chilled_share": chilled,
            "promised_hours": promised, "truck_util": util, "driver_tenure_m": tenure, "ramadan": ram, "pre_eid": pre,
            "delivery_hours": hours, "late": (hours > promised).astype(int),
        })

    a = draw(old, 72000, START, END, False)
    b = draw(new, 8000, NORTH_START, END, True)
    o = pd.concat([a, b]).sort_values(["date", "customer"], kind="stable").reset_index(drop=True)
    o.insert(0, "order", [f"O{i + 1:06d}" for i in range(len(o))])
    return o


# ------------------------------------------------------------------ the vendor claim (Lab 1.4)
def leaderboard(rng) -> pd.DataFrame:
    rows = []
    for i in range(36):
        n = int(np.exp(rng.uniform(np.log(300), np.log(60000))))
        overlap = i % 7 in (0, 3) or i in (5, 22)             # in the model's pretraining or selection suites
        small = n < 3000
        tuned = 0.80 + rng.uniform(-0.08, 0.1)
        edge = (0.012 if small else -0.006) + (0.012 if overlap else 0.0) + rng.normal(0, 0.006)
        pre = tuned + edge
        default = tuned - abs(rng.normal(0.018, 0.008))       # the baseline as the claim ran it
        rows.append({"dataset": f"D{i + 1:02d}", "rows": n, "features": int(rng.integers(6, 120)),
                     "pretrained_auc": round(pre, 4), "gbm_auc_as_claimed": round(default, 4),
                     "gbm_trials_as_claimed": 0, "gbm_auc_tuned": round(tuned, 4), "gbm_trials_tuned": 100,
                     "pretrained_ensemble_members": 8, "in_pretraining_or_selection": int(overlap)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ adoption (Lab 4.4)
def adoption():
    cands = [
        {"task": "late-delivery risk (tables)", "model": "TabICLv2", "kind": "pretrained tabular model",
         "weights_licence": "BSD-3-Clause", "commercial_use": "yes", "outputs_restricted": "no",
         "deployment": ["self-hosted in Bustan's data centre in Riyadh", "Google Colab (Google-run machines, location not chosen by the user)"],
         "source": "https://github.com/soda-inria/tabicl"},
        {"task": "late-delivery risk (tables)", "model": "TabPFN-2.6", "kind": "pretrained tabular model",
         "weights_licence": "TabPFN-2.6 Non-Commercial License", "commercial_use": "no: needs a separate commercial licence",
         "outputs_restricted": "yes: outputs for non-commercial purposes only",
         "deployment": ["self-hosted in Bustan's data centre in Riyadh", "the vendor's hosted service outside the Kingdom"],
         "source": "https://huggingface.co/Prior-Labs/tabpfn_2_6"},
        {"task": "daily demand (series)", "model": "Chronos-2", "kind": "pretrained forecaster",
         "weights_licence": "Apache-2.0", "commercial_use": "yes", "outputs_restricted": "no",
         "deployment": ["self-hosted in Bustan's data centre in Riyadh", "a cloud endpoint in a region outside the Kingdom"],
         "source": "https://huggingface.co/amazon/chronos-2"},
        {"task": "daily demand (series)", "model": "TimesFM 2.5", "kind": "pretrained forecaster",
         "weights_licence": "Apache-2.0", "commercial_use": "yes", "outputs_restricted": "no",
         "deployment": ["self-hosted in Bustan's data centre in Riyadh"],
         "source": "https://huggingface.co/google/timesfm-2.5-200m-pytorch"},
    ]
    register = pd.DataFrame([
        {"dataset": "orders (with customer ids, depot, distance)", "level": "Restricted", "personal_data": "yes",
         "note": "grocery customers include sole traders: a customer id and address identify a person"},
        {"dataset": "orders, aggregated by depot and week", "level": "Public", "personal_data": "no",
         "note": "published in the annual report"},
        {"dataset": "daily demand by product and depot", "level": "Restricted", "personal_data": "no",
         "note": "commercially sensitive: supplier terms can be inferred"},
        {"dataset": "driver records (tenure, routes)", "level": "Restricted", "personal_data": "yes",
         "note": "employee data"},
    ])
    costs = pd.DataFrame([  # synthetic internal rates of Bustan: illustrative, not market prices
        {"option": "self-hosted CPU server (existing)", "sar_per_hour": 4.0},
        {"option": "self-hosted GPU server (to buy or lease)", "sar_per_hour": 38.0},
        {"option": "hosted service, per 1,000 predictions", "sar_per_hour": None, "sar_per_1000": 6.0},
    ])
    return cands, register, costs


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    (DATA / "adoption").mkdir(exist_ok=True)
    cal = calendar()
    cal.assign(date=cal.date.dt.strftime("%Y-%m-%d")).to_csv(DATA / "calendar.csv", index=False, lineterminator="\n")
    prod = products(rng)
    y, meta = demand(rng, cal, prod)
    prod.drop(columns="base").to_csv(DATA / "products.csv", index=False, lineterminator="\n")
    meta.to_csv(DATA / "series_meta.csv", index=False, lineterminator="\n")
    np.savez_compressed(DATA / "demand.npz", y=y.astype("float32"), start=np.array([START.strftime("%Y-%m-%d")]),
                        series=np.array(meta.series, dtype=str))
    o = orders(rng, cal)
    o.to_csv(DATA / "orders.csv.gz", index=False, lineterminator="\n",
             compression={"method": "gzip", "mtime": 0, "compresslevel": 9})
    trucks = {"truck": {"m3": 60.0, "kg": 20000, "sar_per_trip": 2400},
              "reefer": {"m3": 40.0, "kg": 14000, "sar_per_trip": 3200, "available_per_week": 3},
              "depot_inbound_m3_per_week": 520.0, "suppliers": {"S1": "dry goods", "S2": "beverages",
                                                                 "S3": "chilled dairy (reefer only)", "S4": "dates and catering"},
              "note": "synthetic rates of Bustan Food Wholesale, not market prices"}
    (DATA / "trucks.json").write_text(json.dumps(trucks, indent=1) + "\n", encoding="utf-8", newline="\n")
    leaderboard(rng).to_csv(DATA / "leaderboard_claim.csv", index=False, lineterminator="\n")
    cands, register, costs = adoption()
    (DATA / "adoption" / "candidates.json").write_text(json.dumps(cands, indent=1, ensure_ascii=False) + "\n",
                                                        encoding="utf-8", newline="\n")
    register.to_csv(DATA / "adoption" / "classification_register.csv", index=False, lineterminator="\n")
    costs.to_csv(DATA / "adoption" / "cost_sheet.csv", index=False, lineterminator="\n")
    print(f"calendar {len(cal)} days, demand {y.shape}, orders {len(o)} "
          f"(late {o.late.mean():.3f}; north {o[o.region == 'northern'].late.mean():.3f}; "
          f"old {o[o.region != 'northern'].late.mean():.3f})")


if __name__ == "__main__":
    main()
