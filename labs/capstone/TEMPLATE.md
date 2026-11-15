# (Your task): evaluation protocol and adoption decision

**Author:** (name) · **Course:** SDAIA Academy DS-404 · **Date:** (date)

## 1. The decision

Who decides, what is decided, and what a wrong decision costs. One paragraph.

## 2. Protocol (written before the results)

| Item | Choice | Why |
|---|---|---|
| Data and cut | | |
| Deployment shift | | |
| Validation scheme | | which shift it matches |
| Comparison scheme | | the one expected to flatter |
| Baseline and budget | | fits, seconds |
| Candidate 1 | | how it is run, who answered |
| Candidate 2 | | |
| Ensemble | | fitted on which predictions |
| Metrics and intervals | | repeats, bootstrap unit |
| Decision rule | | the gain that would justify adoption |

## 3. Results

Every number with its interval and the command that produced it.

| Model | Metric (interval) | Run time | Command |
|---|---|---|---|
| Baseline | | | |
| Candidate 1 | | | |
| Candidate 2 | | | |
| Ensemble | | | |

### Validation under shift

| Scheme | Estimate | Deployed or held-out shift | Gap |
|---|---|---|---|

### Subgroups

| Group | Metric | Size |
|---|---|---|

### Decision cost (where a decision follows)

| Model | Regret | Error | Rank by regret | Rank by error |
|---|---|---|---|---|

## 4. Adoption decision

| Candidate | Licence (checked on, source) | Data level and residency | Cost per month | Decision | Conditions |
|---|---|---|---|---|---|

Memo: three lines for the decision owner.

## 5. Reproduce

```
uv sync
uv run python (script) --seed 404
```

Versions: Python, packages (from `uv.lock`), model versions.

## 6. Verification note on AI-assisted steps

| Step | What the assistant produced | How it was checked |
|---|---|---|
