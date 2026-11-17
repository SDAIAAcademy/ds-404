---
course: "Advanced Modelling for Structured Data"
code: "DS-404"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-404/index.html"
---

# Advanced Modelling for Structured Data

النمذجة المتقدمة للبيانات المهيكلة

Data Science Track · 4 teaching days + capstone day · 26 hours · Level 400

Decide when pretrained models, stacked ensembles and decision-aware pipelines earn their place, measured against tuned baselines and under shift

Participants decide when pretrained models, ensembles and decision-aware pipelines earn their place in structured-data work. On a fictional food wholesaler's synthetic orders and demand, they test pretrained tabular models and zero-shot forecasters against tuned baselines, stack ensembles, turn forecasts into orders by optimisation, score models by decision cost and validate under the shift of a new depot region. They end with an adoption decision weighing accuracy, licence, cost and data residency.

**Who it is for:** Specialists. Data scientists who have completed the level 300 core or can show equivalent work: a defended analysis with a validated model or a causal estimate that they can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Bustan Food Wholesale.** Bustan Food Wholesale is a fictional food wholesaler with depots in Riyadh, Buraidah, Jeddah and Dammam and a new northern region around Hail and Tabuk. Participants test a pretrained tabular model against a tuned boosted-tree baseline on 80,000 orders, score zero-shot forecasters with and without Ramadan and Eid covariates on 400 daily demand series over three Hijri years, stack forecasters, turn forecasts into orders and truck plans by optimisation, rank forecasters by the regret they cause, validate a model under the shift of the new region and end with an adoption decision on licence, cost and data residency. All data is synthetic and generated with a fixed seed.

## The course

### Day 1: Pretrained models for tables

Slides: https://sdaiaacademy.github.io/ds-404/slides/day1.md

- **Session 1: Tuned boosted trees first**
  - State a tuning budget in fits and seconds before comparing models
  - Tune a boosted-tree model by random search under grouped cross-validation
  - Explain why random K-fold flatters a model when customers repeat
  - [Lab 1.1: A tuned baseline under a budget (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D1_1_tune_boosted_tree.ipynb)
- **Session 2: In-context learning on tables**
  - Describe what a tabular foundation model learns in pretraining and what it does at prediction
  - Run a pretrained tabular model on the same folds as the baseline
  - Compare accuracy and run time with the tuning time included
  - [Lab 1.2: A pretrained model on the same folds (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D1_2_run_pretrained_tabular.ipynb)
- **Session 3: Learning curves decide**
  - Build learning curves from repeated subsamples at six training sizes
  - Put an interval on each point and read where the curves cross
  - State where each approach wins and where the evidence is too thin to say
  - [Lab 1.3: Learning curves with intervals (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D1_3_plot_both_across.ipynb)
- **Session 4: Reading a leaderboard claim**
  - Find unequal tuning budgets in a benchmark table
  - Remove datasets the pretrained model may have seen or been selected on
  - Recount wins as a paired difference with an interval
  - [Lab 1.4: Check a leaderboard claim (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D1_4_check_claim_unequal.ipynb)

### Day 2: Forecasting with pretrained models

Slides: https://sdaiaacademy.github.io/ds-404/slides/day2.md

- **Session 1: Zero-shot against a benchmark floor**
  - Run a zero-shot forecaster on 400 series at four origins
  - Score scaled point error and quantile loss on the same windows as supplied benchmarks
  - Report each method against the seasonal naive floor
  - [Lab 2.1: Zero-shot against the supplied backtests (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D2_1_score_zero_shot.ipynb)
- **Session 2: Where zero-shot loses**
  - Classify series as smooth, intermittent or short from their own history
  - Compare methods by series group, point and quantile
  - Turn a group table into a routing rule
  - [Lab 2.2: Where the benchmark wins (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D2_2_find_series_groups.ipynb)
- **Session 3: Holidays that move**
  - Show why a yearly lag must respect both the week and the Hijri year
  - Supply Ramadan and Eid as known future covariates
  - Measure error in holiday weeks separately from ordinary weeks
  - [Lab 2.3: Ramadan and Eid as covariates (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D2_3_add_ramadan_eid.ipynb)
- **Session 4: Ensembles worth their cost**
  - Build a stacking set from rolling-origin forecasts without leakage
  - Fit non-negative stacking weights and test them on later windows
  - Report the improvement over the best single model with a bootstrap interval
  - [Lab 2.4: Stacking on rolling-origin windows (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D2_4_stack_forecasters_rolling.ipynb)

### Day 3: From forecast to order

Slides: https://sdaiaacademy.github.io/ds-404/slides/day3.md

- **Session 1: Asymmetric costs**
  - Compute underage and overage costs from a cost table
  - Turn a critical ratio into an order quantity from quantile forecasts
  - Measure the cost of ordering the median instead
  - [Lab 3.1: Newsvendor quantities (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D3_1_compute_newsvendor_quantities.ipynb)
- **Session 2: Optimising under capacity**
  - Write an order and truck plan as variables, constraints and an objective
  - Solve it with an open solver and read the solution
  - Find which constraint binds and what relaxing it is worth
  - [Lab 3.2: Orders and trucks for one depot week (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D3_2_solve_order_truck.ipynb)
- **Session 3: Scoring by regret**
  - Compute the realised cost of a plan made from each forecaster
  - Compute regret against the plan made with hindsight
  - Compare the ranking by regret with the ranking by error
  - [Lab 3.3: Regret against error (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D3_3_rank_three_forecasters.ipynb)
- **Session 4: Uncertainty in the plan**
  - Simulate weekly demand scenarios with correlated days and products
  - Report shortage probability and fill rate of a fixed plan
  - Show how independence assumptions understate shortage risk
  - [Lab 3.4: Stress-test the plan (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D3_4_stress_test_plan.ipynb)

### Day 4: Shift and adoption

Slides: https://sdaiaacademy.github.io/ds-404/slides/day4.md

- **Session 1: Detecting shift**
  - Name covariate, label and concept shift and a test for each
  - Run adversarial validation between training and deployment data
  - Read which features carry the shift
  - [Lab 4.1: Adversarial validation (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D4_1_run_adversarial_validation.ipynb)
- **Session 2: Validating under shift**
  - Run random, grouped, temporal and region-held-out validation on one model
  - Compare each estimate with the error in the new region
  - Report error by region and customer segment before and after shift
  - [Lab 4.2: Four schemes against deployed error (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D4_2_compare_random_grouped.ipynb)
- **Session 3: Correcting for shift**
  - Recall split conformal intervals and the exchangeability they assume
  - Estimate density-ratio weights from an adversarial classifier
  - Apply weighted conformal intervals and check coverage in the new region
  - [Lab 4.3: Weighted conformal intervals (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D4_3_apply_weighted_conformal.ipynb)
- **Session 4: Adoption conditions**
  - Check the licence of a model's weights and outputs against the intended use
  - Apply the classification level and the transfer rules to a deployment option
  - Write an adoption decision with its conditions and choose the capstone task
  - [Lab 4.4: Licence, cost and residency (25 min)](https://sdaiaacademy.github.io/ds-404/labs/D4_4_complete_licence_cost.ipynb)

### Capstone: Capstone project presentations

Slides: https://sdaiaacademy.github.io/ds-404/slides/capstone.md

## Capstone

An evaluation protocol and adoption decision for one structured-data task (two pretrained candidates, a tuned baseline, an ensemble, validation under a stated shift, regret where relevant) in a public repository whose README reports measured results; defence before two assessors

**What you submit:** A public GitHub repository link: the evaluation protocol run end to end (two pretrained candidates, a tuned baseline, an ensemble, validation under a stated shift, regret where a decision follows), a README that reports the measured results with intervals and the adoption decision with its conditions, the run logs and a verification note on AI-assisted steps. Then a defence before two assessors: one assessor names a new shift scenario and the participant predicts which validation result changes before re-running it.

**Rubric:**

- Fair comparison: 25%
- Validation under shift: 25%
- Ensemble or decision analysis: 20%
- Adoption reasoning: 15%
- Reproducibility: 15%

## Outcomes

- **LO1** Evaluate a pretrained tabular model against a tuned, ensembled boosted-tree baseline across training-set sizes, given late-delivery records from 500 to 50,000 orders and a fixed compute budget (Performance) Evidence: learning curves with intervals showing where each approach wins
- **LO2** Evaluate zero-shot pretrained forecasters, with and without covariates, against statistical and tuned learned benchmarks by rolling-origin backtests, given 400 demand series with Hijri holiday covariates (Performance) Evidence: point and quantile accuracy by series group
- **LO3** Implement a stacked ensemble from out-of-fold or rolling-origin predictions and measure its improvement over the best single model, given four fitted candidate models (Applied) Evidence: a stacking run with the improvement and its interval
- **LO4** Build a predict-then-optimise pipeline and judge forecasters by the decision cost they cause, given capacity limits, holding costs and shortage costs (Performance) Evidence: regret per forecaster compared with the error ranking
- **LO5** Diagnose distribution shift between training and deployment data and validate a model under it, given a new depot region and a holiday season that moves between years (Performance) Evidence: a shift report with adversarial, grouped and temporal validation and a corrected error estimate
- **LO6** Decide whether to adopt a pretrained model for one task, weighing accuracy, cost, licence and data residency, given two candidate models, their licences and the wholesaler's data classification (Performance) Evidence: a defended adoption memo with a reusable evaluation protocol

## Tools

- **Pretrained tabular models:** TabICL, TabPFN family (non-commercial weights)
- **Pretrained forecasters:** Chronos family, TimesFM
- **Baselines and ensembling:** LightGBM, StatsForecast, AutoGluon
- **Optimisation modelling and solvers:** Pyomo or PuLP with HiGHS, OR-Tools
- **Conformal prediction:** MAPIE, course-provided implementations

## Prerequisites

- DS-304 Forecasting and Time Series
- DS-305 Uncertainty, Interpretability and Model Validation
- Tunes boosted-tree models with cross-validation
- Backtests forecasters with rolling-origin cross-validation
- Formulates a simple linear programme, or completes the pre-work
- Setup: a laptop CPU with uv, or Google Colab. No key and no model download is needed on a laptop: the pretrained models run in a Colab runtime, and a labelled stand-in answers elsewhere. Run 00_setup_check before Day 1; see labs/README.md.

## Sources

### Pretrained tabular models and their licences

- [Qu and others: TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564)
- [Qu and others: TabICLv2: A better, faster, scalable, and open tabular foundation model](https://arxiv.org/abs/2602.11139)
- [TabICL repository and README (BSD-3-Clause)](https://github.com/soda-inria/tabicl)
- [TabPFN-2.6: model card](https://huggingface.co/Prior-Labs/tabpfn_2_6)
- [TabPFN-2.6: licence file (non-commercial)](https://huggingface.co/Prior-Labs/tabpfn_2_6/raw/main/LICENSE)
- [TabPFN-2.5: licence file (non-commercial)](https://huggingface.co/Prior-Labs/tabpfn_2_5/raw/main/LICENSE)

### Pretrained forecasters and their licences

- [Ansari and others: Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821)
- [Chronos-2: model card (Apache-2.0)](https://huggingface.co/amazon/chronos-2)
- [chronos-forecasting repository and README](https://github.com/amazon-science/chronos-forecasting)
- [Das and others: A decoder-only foundation model for time-series forecasting (TimesFM)](https://arxiv.org/abs/2310.10688)
- [TimesFM 2.5: model card (Apache-2.0)](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [TimesFM repository](https://github.com/google-research/timesfm)

### Baselines, evaluation and ensembles

- [Bergstra and Bengio (2012): Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html)
- [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/)
- [Hyndman and Athanasopoulos: FPP3, evaluating point forecast accuracy (scaled errors)](https://otexts.com/fpp3/accuracy.html)
- [Hyndman and Athanasopoulos: FPP3, forecast combinations](https://otexts.com/fpp3/combinations.html)

### Decisions and optimisation

- [Elmachtoub and Grigas: Smart "Predict, then Optimize"](https://arxiv.org/abs/1710.08005)
- [PuLP on PyPI](https://pypi.org/project/PuLP/)
- [HiGHS: high performance software for linear optimisation](https://highs.dev/)

### Shift and conformal prediction

- [Tibshirani, Barber, Candès and Ramdas (2019): Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019)
- [MAPIE documentation](https://mapie.readthedocs.io/en/stable/)

### National references

- [National Data Governance Policies, including the Data Classification Policy (NDMO)](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [Regulation on Personal Data Transfer outside the Kingdom](https://sdaia.gov.sa/Documents/RegulationonPersonalDataEN.pdf)
- [AI Ethics Principles](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)

---

SDAIA Academy · Data Science Track · Advanced Modelling for Structured Data

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit 49dbc1a
