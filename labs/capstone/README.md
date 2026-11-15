# Capstone starter

You write an evaluation protocol and an adoption decision for **one** structured-data task, run it end to end, and defend it: 10 minutes, then 10 minutes of questions from two assessors. One assessor names a new shift scenario; you predict which validation result changes before you re-run it.

## Choose the task

You chose it at the end of Lab 4.4 (`CAPSTONE_TASK`):

| Task | Data | Pretrained candidates | Tuned baseline | Shift to validate under | Decision to price |
|---|---|---|---|---|---|
| Late-delivery risk (tables) | `data/orders.csv.gz` | two tabular models (in the course: TabICLv2, and a second of your choice whose licence you check) | LightGBM under a stated budget | the northern region, or a shift you justify | which orders to warn dispatchers about, or the promised hours |
| Daily demand (series) | `data/demand.npz` | two forecasters (in the course: Chronos-2 and TimesFM 2.5) | the supplied LightGBM backtests or your own | holiday windows, short series, or one depot held out (the demand series have no northern depot) | the weekly orders of Day 3 |

You may use another public structured dataset instead, if you state its shift and its decision as clearly. Never use real customer or personal data.

## What the repository must contain

1. **The protocol**, written before the results: the data cut, the validation scheme and the shift it matches, the baseline and its tuning budget, the candidates and how they are run, the metrics and how their intervals are computed, the ensemble, and the decision rule.
2. **The runs**: code that reproduces every number from a clean clone (`uv sync`, then one command per result), with the logs it produced.
3. **The results**, each with its interval and the run that produced it: candidates against the baseline, the ensemble's gain over the best single model, validation under the shift, error by subgroup, and regret where a decision follows.
4. **The adoption decision**: go, go with conditions, not now or no-go, with the licence, data residency and cost checks of Lab 4.4.
5. **A verification note** on every step where an AI assistant helped: what it produced and how you checked it.

Copy `TEMPLATE.md` to your repository as its `README.md` and fill it in.

## How it is scored

| Criterion | Weight | Full marks look like |
|---|---|---|
| Fair comparison | 25% | same folds or windows for every model; the baseline tuned under a budget you state; run time counted on both sides |
| Validation under shift | 25% | a scheme that matches the deployment, compared with at least one that does not; a reasoned prediction for the assessors' scenario |
| Ensemble or decision analysis | 20% | an ensemble fitted without leakage with its gain and interval, or regret per model against the error ranking |
| Adoption reasoning | 15% | licence, residency, cost and subgroup error, each with its source and date |
| Reproducibility | 15% | a clean clone reproduces the numbers; seeds and versions pinned |

## Starting points in this repository

- `labs/bustan.py`: loaders, grouped and random folds, the tuned-ensemble procedure, bootstrap intervals, the scoring functions and the zero-shot helper.
- `labs/solutions/`: every lab as the course ran it. Reuse the code; cite which lab it came from.
- `labs/data/adoption/`: the candidate cards and the classification register. Re-check every licence on the model's official page on the day you submit, and write the date.
