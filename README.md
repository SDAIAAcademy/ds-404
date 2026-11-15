<!-- academy:title -->
# Advanced Modelling for Structured Data

النمذجة المتقدمة للبيانات المهيكلة

Data Science Track · SDAIA Academy · DS-404 · Level 400

REPLACE: one line under the title: what participants can do after the course

Participants decide when pretrained models, ensembles and decision-aware pipelines earn their place in structured-data work. On a fictional food wholesaler's synthetic orders and demand, they test pretrained tabular models and zero-shot forecasters against tuned baselines, stack ensembles, turn forecasts into orders by optimisation, score models by decision cost and validate under the shift of a new depot region. They end with an adoption decision weighing accuracy, licence, cost and data residency.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ds-404/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Evaluate a pretrained tabular model against a tuned, ensembled boosted-tree baseline across training-set sizes, given late-delivery records from 500 to 50,000 orders and a fixed compute budget | Performance | learning curves with intervals showing where each approach wins |
| LO2 | Evaluate zero-shot pretrained forecasters, with and without covariates, against statistical and tuned learned benchmarks by rolling-origin backtests, given 400 demand series with Hijri holiday covariates | Performance | point and quantile accuracy by series group |
| LO3 | Implement a stacked ensemble from out-of-fold or rolling-origin predictions and measure its improvement over the best single model, given four fitted candidate models | Applied | a stacking run with the improvement and its interval |
| LO4 | Build a predict-then-optimise pipeline and judge forecasters by the decision cost they cause, given capacity limits, holding costs and shortage costs | Performance | regret per forecaster compared with the error ranking |
| LO5 | Diagnose distribution shift between training and deployment data and validate a model under it, given a new depot region and a holiday season that moves between years | Performance | a shift report with adversarial, grouped and temporal validation and a corrected error estimate |
| LO6 | Decide whether to adopt a pretrained model for one task, weighing accuracy, cost, licence and data residency, given two candidate models, their licences and the wholesaler's data classification | Performance | a defended adoption memo with a reusable evaluation protocol |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Pretrained models for tables | Tuned boosted trees first; In-context learning on tables; Learning curves decide; Reading a leaderboard claim | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Forecasting with pretrained models | Zero-shot against a benchmark floor; Where zero-shot loses; Holidays that move; Ensembles worth their cost | [`slides/day2.html`](slides/day2.html) |
| Day 3 | From forecast to order | Asymmetric costs; Optimising under capacity; Scoring by regret; Uncertainty in the plan | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Shift and adoption | Detecting shift; Validating under shift; Correcting for shift; Adoption conditions | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Tune a boosted-tree baseline under a fixed budget with grouped cross-validation | [`D1_1_tune_boosted_tree.ipynb`](labs/D1_1_tune_boosted_tree.ipynb) |
| 1.2 | Day 1 · Session 2 | Run a pretrained tabular model on the same splits; compare accuracy and run time | [`D1_2_run_pretrained_tabular.ipynb`](labs/D1_2_run_pretrained_tabular.ipynb) |
| 1.3 | Day 1 · Session 3 | Plot both across six training sizes, using supplied scores for the largest | [`D1_3_plot_both_across.ipynb`](labs/D1_3_plot_both_across.ipynb) |
| 1.4 | Day 1 · Session 4 | Check a claim for unequal tuning budgets and pretraining overlap | [`D1_4_check_claim_unequal.ipynb`](labs/D1_4_check_claim_unequal.ipynb) |
| 2.1 | Day 2 · Session 1 | Score zero-shot forecasts for 400 series against supplied benchmark backtests | [`D2_1_score_zero_shot.ipynb`](labs/D2_1_score_zero_shot.ipynb) |
| 2.2 | Day 2 · Session 2 | Find the series groups, such as intermittent items, where the benchmark wins | [`D2_2_find_series_groups.ipynb`](labs/D2_2_find_series_groups.ipynb) |
| 2.3 | Day 2 · Session 3 | Add Ramadan and Eid covariates and measure holiday-week error | [`D2_3_add_ramadan_eid.ipynb`](labs/D2_3_add_ramadan_eid.ipynb) |
| 2.4 | Day 2 · Session 4 | Stack forecasters on rolling-origin windows; test the improvement against its interval | [`D2_4_stack_forecasters_rolling.ipynb`](labs/D2_4_stack_forecasters_rolling.ipynb) |
| 3.1 | Day 3 · Session 1 | Compute newsvendor quantities from quantile forecasts for 20 products | [`D3_1_compute_newsvendor_quantities.ipynb`](labs/D3_1_compute_newsvendor_quantities.ipynb) |
| 3.2 | Day 3 · Session 2 | Solve an order and truck-allocation programme for one depot week | [`D3_2_solve_order_truck.ipynb`](labs/D3_2_solve_order_truck.ipynb) |
| 3.3 | Day 3 · Session 3 | Rank three forecasters by regret and by error | [`D3_3_rank_three_forecasters.ipynb`](labs/D3_3_rank_three_forecasters.ipynb) |
| 3.4 | Day 3 · Session 4 | Stress-test the plan with simulated demand; report shortage risk | [`D3_4_stress_test_plan.ipynb`](labs/D3_4_stress_test_plan.ipynb) |
| 4.1 | Day 4 · Session 1 | Run adversarial validation between old depots and the new region | [`D4_1_run_adversarial_validation.ipynb`](labs/D4_1_run_adversarial_validation.ipynb) |
| 4.2 | Day 4 · Session 2 | Compare random, grouped and temporal validation with deployed error | [`D4_2_compare_random_grouped.ipynb`](labs/D4_2_compare_random_grouped.ipynb) |
| 4.3 | Day 4 · Session 3 | Apply weighted conformal intervals and check coverage in the new region | [`D4_3_apply_weighted_conformal.ipynb`](labs/D4_3_apply_weighted_conformal.ipynb) |
| 4.4 | Day 4 · Session 4 | Complete a licence, cost and residency table; choose the capstone task | [`D4_4_complete_licence_cost.ipynb`](labs/D4_4_complete_licence_cost.ipynb) |
<!-- /academy:labs -->

## Capstone

An evaluation protocol and adoption decision for one structured-data task (two pretrained candidates, a tuned baseline, an ensemble, validation under a stated shift, regret where relevant) in a public repository whose README reports measured results; defence before two assessors

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- DS-304 Forecasting and Time Series
- DS-305 Uncertainty, Interpretability and Model Validation
- Tunes boosted-tree models with cross-validation
- Backtests forecasters with rolling-origin cross-validation
- Formulates a simple linear programme, or completes the pre-work

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Pretrained tabular models:** TabICL, TabPFN family (non-commercial weights)
- **Pretrained forecasters:** Chronos family, TimesFM (non-commercial weights)
- **Baselines and ensembling:** LightGBM, StatsForecast, AutoGluon
- **Optimisation modelling and solvers:** Pyomo or PuLP with HiGHS, OR-Tools
- **Conformal prediction:** MAPIE, course-provided implementations
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- NDMO Data Classification Policy: REPLACE: the address of the official text
- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- Regulation on Personal Data Transfer: REPLACE: the address of the official text
- SDAIA AI Ethics Principles: REPLACE: the address of the official text
