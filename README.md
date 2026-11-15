<!-- academy:title -->
# Advanced Modelling for Structured Data

النمذجة المتقدمة للبيانات المهيكلة

Data Science Track · SDAIA Academy · DS-404 · Level 400

Decide when pretrained models, stacked ensembles and decision-aware pipelines earn their place, measured against tuned baselines and under shift

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

**Bustan Food Wholesale** (بستان لتجارة المواد الغذائية بالجملة) is a fictional food wholesaler with depots in Riyadh, Buraidah, Jeddah and Dammam and a northern region around Hail and Tabuk opened in August 2025. Every lab works on its synthetic data, generated with a fixed seed by `labs/prepare_data.py`: 80,000 orders with delivery hours, 400 daily demand series over three Hijri years with Ramadan and both Eids, a cost and truck table, a vendor benchmark claim and the adoption files (candidate licences, classification register, cost sheet).

People in the case: Ahlam Al-Kuwaileet (head of supply planning, the decision owner), Shadi Al-Baiz (operations manager of the northern region), Imad Al-Sharqi (fleet and depot capacity planner), Asil Al-Rajeh (data protection officer), Rafif Al-Moaibed (analyst who forwarded the vendor claim) and Sarab Al-Mashouq (senior data scientist: the participant's role).

**Pretrained models in the labs.** The real models (TabICL, Chronos-2) run only in a Google Colab runtime, where their weights download; nothing is downloaded to a laptop and no API key is needed. Everywhere else a labelled stand-in answers, and every result says which one did. The course's own reference results were written by the stand-ins.

## How the course is organised

- **Pre-work, before Day 1 (about three hours).** Quantile (pinball) loss, newsvendor logic and simple linear programmes, with an entry quiz: the four practice questions that start with "Pre-work." on the course's practice page. Run `00_setup_check.ipynb`, which also solves a small integer programme with HiGHS. Day 2 checks the pinball loss and Day 3 the newsvendor result and the programme instead of teaching them again.
- **Every lab starts from a supplied reference state**: the data files and, in `labs/data/reference/`, the results earlier labs would have produced. No lab depends on a participant's earlier run, and every exercise has a safety net.
- **AI policy.** AI off for the supervised check and for the Day 1 baseline (Lab 1.1). In the other labs an assistant may give hints, not finished pipelines. In the capstone any use of AI is allowed, with a verification note on assisted steps and a live defence.
- **Responsible use.** Comparisons report error by depot region and customer segment (by depot for the demand series), and Day 4 compares them before and after the shift to the northern region, so that a model that serves new or remote depots worse is caught before adoption. The Day 4 adoption lab applies Bustan's classification levels and the personal data transfer rules before a hosted model may receive its records.
- **Assessment.** An AI-off supervised check (choose a validation scheme for a described shift, compute a newsvendor quantity, read a learning curve) 20%; the capstone rubric 60%; the defence before two assessors 20%. Passing: 70% overall, with at least 60% on the capstone rubric and on the supervised check.
- **Follow-up, within three weeks of the course.** Apply the evaluation protocol to one workplace or public dataset and report the measured results to the trainer.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | A tuned baseline under a budget | [`D1_1_tune_boosted_tree.ipynb`](labs/D1_1_tune_boosted_tree.ipynb) |
| 1.2 | Day 1 · Session 2 | A pretrained model on the same folds | [`D1_2_run_pretrained_tabular.ipynb`](labs/D1_2_run_pretrained_tabular.ipynb) |
| 1.3 | Day 1 · Session 3 | Learning curves with intervals | [`D1_3_plot_both_across.ipynb`](labs/D1_3_plot_both_across.ipynb) |
| 1.4 | Day 1 · Session 4 | Check a leaderboard claim | [`D1_4_check_claim_unequal.ipynb`](labs/D1_4_check_claim_unequal.ipynb) |
| 2.1 | Day 2 · Session 1 | Zero-shot against the supplied backtests | [`D2_1_score_zero_shot.ipynb`](labs/D2_1_score_zero_shot.ipynb) |
| 2.2 | Day 2 · Session 2 | Where the benchmark wins | [`D2_2_find_series_groups.ipynb`](labs/D2_2_find_series_groups.ipynb) |
| 2.3 | Day 2 · Session 3 | Ramadan and Eid as covariates | [`D2_3_add_ramadan_eid.ipynb`](labs/D2_3_add_ramadan_eid.ipynb) |
| 2.4 | Day 2 · Session 4 | Stacking on rolling-origin windows | [`D2_4_stack_forecasters_rolling.ipynb`](labs/D2_4_stack_forecasters_rolling.ipynb) |
| 3.1 | Day 3 · Session 1 | Newsvendor quantities | [`D3_1_compute_newsvendor_quantities.ipynb`](labs/D3_1_compute_newsvendor_quantities.ipynb) |
| 3.2 | Day 3 · Session 2 | Orders and trucks for one depot week | [`D3_2_solve_order_truck.ipynb`](labs/D3_2_solve_order_truck.ipynb) |
| 3.3 | Day 3 · Session 3 | Regret against error | [`D3_3_rank_three_forecasters.ipynb`](labs/D3_3_rank_three_forecasters.ipynb) |
| 3.4 | Day 3 · Session 4 | Stress-test the plan | [`D3_4_stress_test_plan.ipynb`](labs/D3_4_stress_test_plan.ipynb) |
| 4.1 | Day 4 · Session 1 | Adversarial validation | [`D4_1_run_adversarial_validation.ipynb`](labs/D4_1_run_adversarial_validation.ipynb) |
| 4.2 | Day 4 · Session 2 | Four schemes against deployed error | [`D4_2_compare_random_grouped.ipynb`](labs/D4_2_compare_random_grouped.ipynb) |
| 4.3 | Day 4 · Session 3 | Weighted conformal intervals | [`D4_3_apply_weighted_conformal.ipynb`](labs/D4_3_apply_weighted_conformal.ipynb) |
| 4.4 | Day 4 · Session 4 | Licence, cost and residency | [`D4_4_complete_licence_cost.ipynb`](labs/D4_4_complete_licence_cost.ipynb) |
<!-- /academy:labs -->

## Capstone

An evaluation protocol and adoption decision for one structured-data task (two pretrained candidates, a tuned baseline, an ensemble, validation under a stated shift, regret where relevant) in a public repository whose README reports measured results; defence before two assessors

**Submission:** a public GitHub repository link: the evaluation protocol run end to end, a README that reports the measured results with intervals and the adoption decision with its conditions, the run logs and a verification note on AI-assisted steps. Start from `labs/capstone/`. The defence is 10 minutes, then 10 minutes of questions from two assessors, one of whom names a new shift scenario.

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
- **Pretrained forecasters:** Chronos family, TimesFM
- **Baselines and ensembling:** LightGBM, StatsForecast, AutoGluon
- **Optimisation modelling and solvers:** Pyomo or PuLP with HiGHS, OR-Tools
- **Conformal prediction:** MAPIE, course-provided implementations
<!-- /academy:tools -->

<!-- academy:sources -->
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
<!-- /academy:sources -->

### National references

- NDMO National Data Governance Policies, including the Data Classification Policy: https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf
- Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf, and its Implementing Regulation: https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf
- Regulation on Personal Data Transfer outside the Kingdom: https://sdaia.gov.sa/Documents/RegulationonPersonalDataEN.pdf
- SDAIA AI Ethics Principles: https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf

**The transfer rules in brief (Day 4, Session 4).** Article 29 of the Personal Data Protection Law allows personal data to be transferred outside the Kingdom only for the purposes it lists, with a level of protection at least equivalent to the Law's and limited to the minimum data needed. The Regulation on Personal Data Transfer outside the Kingdom requires a risk assessment before a transfer in the cases of its Article 7 (purpose and legal basis, nature and scope, safeguards, data minimisation, possible effects, mitigations) and, in the exempted cases of its Article 4, appropriate safeguards: standard contractual clauses, binding common rules or a certificate of accreditation. The National Data Governance Policies set four classification levels (Top Secret, Secret, Restricted, Public) for data held by public entities; Bustan, a private company, applies the same levels in its own register. This summary is a teaching aid, not legal advice: read the official texts.
