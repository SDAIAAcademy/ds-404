---
course: "Advanced Modelling for Structured Data"
code: "DS-404"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-404/"
---

# Advanced Modelling for Structured Data

Decide when pretrained models, stacked ensembles and decision-aware pipelines earn their place, measured against tuned baselines and under shift

Participants decide when pretrained models, ensembles and decision-aware pipelines earn their place in structured-data work. On a fictional food wholesaler's synthetic orders and demand, they test pretrained tabular models and zero-shot forecasters against tuned baselines, stack ensembles, turn forecasts into orders by optimisation, score models by decision cost and validate under the shift of a new depot region. They end with an adoption decision weighing accuracy, licence, cost and data residency.

- **Audience:** Specialists. Data scientists who have completed the level 300 core or can show equivalent work: a defended analysis with a validated model or a causal estimate that they can explain.
- **Prerequisites:** DS-304 Forecasting and Time Series; DS-305 Uncertainty, Interpretability and Model Validation; Tunes boosted-tree models with cross-validation; Backtests forecasters with rolling-origin cross-validation; Formulates a simple linear programme, or completes the pre-work; Setup: a laptop CPU with uv, or Google Colab. No key and no model download is needed on a laptop: the pretrained models run in a Colab runtime, and a labelled stand-in answers elsewhere. Run 00_setup_check before Day 1; see labs/README.md.
- **Case study:** Bustan Food Wholesale: Bustan Food Wholesale is a fictional food wholesaler with depots in Riyadh, Buraidah, Jeddah and Dammam and a new northern region around Hail and Tabuk. Participants test a pretrained tabular model against a tuned boosted-tree baseline on 80,000 orders, score zero-shot forecasters with and without Ramadan and Eid covariates on 400 daily demand series over three Hijri years, stack forecasters, turn forecasts into orders and truck plans by optimisation, rank forecasters by the regret they cause, validate a model under the shift of the new region and end with an adoption decision on licence, cost and data residency. All data is synthetic and generated with a fixed seed.

## Outcomes

- LO1 (Performance): Evaluate a pretrained tabular model against a tuned, ensembled boosted-tree baseline across training-set sizes, given late-delivery records from 500 to 50,000 orders and a fixed compute budget
- LO2 (Performance): Evaluate zero-shot pretrained forecasters, with and without covariates, against statistical and tuned learned benchmarks by rolling-origin backtests, given 400 demand series with Hijri holiday covariates
- LO3 (Applied): Implement a stacked ensemble from out-of-fold or rolling-origin predictions and measure its improvement over the best single model, given four fitted candidate models
- LO4 (Performance): Build a predict-then-optimise pipeline and judge forecasters by the decision cost they cause, given capacity limits, holding costs and shortage costs
- LO5 (Performance): Diagnose distribution shift between training and deployment data and validate a model under it, given a new depot region and a holiday season that moves between years
- LO6 (Performance): Decide whether to adopt a pretrained model for one task, weighing accuracy, cost, licence and data residency, given two candidate models, their licences and the wholesaler's data classification

## Day 1: Pretrained models for tables

A tuned baseline, a pretrained model and the curve between them

#### [day1#title] Pretrained models for tables

*SDAIA Academy · Advanced Modelling for Structured Data*

Day 1: A tuned baseline, a pretrained model and the curve between them

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-404

#### [day1#four-days-one-decision] Four days of evidence, then one adoption decision you defend

*The course at a glance*

- **Day 1 · Tables.** A tuned baseline, a pretrained tabular model, learning curves, a leaderboard claim.
- **Day 2 · Series.** Zero-shot forecasts on 400 series, by group, with Ramadan and Eid, and stacking.
- **Day 3 · Decisions.** Quantiles into orders and trucks; forecasters ranked by the regret they cause.
- **Day 4 · Shift and adoption.** A new region: shift found, validated, corrected; licence, cost and residency.

> **Day 5 · Capstone.** An evaluation protocol and an adoption decision for one task, in a public repository, defended before two assessors.

#### [day1#evidence-before-adoption] Every new model meets a fair baseline, the same splits and an interval

*How this course works*

- **Fair.** Same data, same folds, same metric, and the tuning budget written down.
- **Sized.** A difference without an interval is an anecdote.
- **Decided.** Accuracy is one column. Cost, licence and the data's classification are the others.

> **AI policy.** AI off for the supervised checks and for the Day 1 baseline (Lab 1.1). In the other labs an assistant may give hints, not finished pipelines. In the capstone, anything goes, with a verification note and a live defence.

#### [day1#bustan-food-wholesale] Bustan Food Wholesale wants to know which new models earn their place

*One case study all week*

A fictional food wholesaler with depots in Riyadh, Jeddah, Dammam and Buraidah, and a new northern region around Hail and Tabuk.

- **80,000 orders.** Over three Hijri years, with delivery hours and a late flag. 8,000 of them come from the new region.
- **400 demand series.** 100 products at four depots, daily, with Ramadan and both Eids from the Umm al-Qura calendar.
- **Costs and capacity.** Margins, holding and spoilage, trucks, reefers and depot space.

#### [day1#where-models-run] Pretrained models run in Colab; on a laptop a labelled stand-in answers

*Lab environment*

1. Run `00_setup_check.ipynb` before Session 1.
2. In **Colab**, the first cell installs the model's package and its weights download into the Colab runtime. Nothing downloads to your laptop.
3. Elsewhere a **stand-in** answers. Every result prints which one answered.
4. Every lab starts from supplied files. No lab needs another lab's output.

```bash
# file: terminal (local option)
git clone https://github.com/SDAIAAcademy/ds-404
cd ds-404/labs
uv sync
uv run jupyter lab
```

#### [day1#agenda] Pretrained models for tables

*Today*

- **Session 1 · Tuned boosted trees first**: A pretrained model is judged against a baseline tuned under a stated budget and validated by customer groups
- **Session 2 · In-context learning on tables**: A tabular foundation model reads the training rows as context and predicts without training on them
- **Session 3 · Learning curves decide**: One score is one point on a curve: compare models across training sizes with intervals
- **Session 4 · Reading a leaderboard claim**: A benchmark win counts only at equal tuning budgets, without pretraining overlap and with an interval

### Session 1: Tuned boosted trees first

#### [day1#s1] Tuned boosted trees first

*Session 1*

- State a tuning budget in fits and seconds before comparing models
- Tune a boosted-tree model by random search under grouped cross-validation
- Explain why random K-fold flatters a model when customers repeat

#### [day1#new-model-old-baseline] "Is the new model better?" means better than what, tuned how much?

*What Ahlam asks*

**Ahlam:** A vendor says its pretrained model predicts late deliveries better than anything we have. Should we switch?

**Sarab:** Better than our boosted trees after we tune them with the same effort, on our own customers. Give me today.

- **The target.** Late: delivered after the promised hours. 18% of orders in the established regions.
- **The trap.** An untuned baseline makes any new model look good.

#### [day1#budget-before-search] Write the tuning budget down before you search

*Before any search*

| Budget item | Lab 1.1 | Why it is fixed first |
| --- | --- | --- |
| Configurations | 20, drawn at random | More draws always find a luckier one |
| Folds | 5, grouped by customer | The estimate the search optimises |
| Fits | 100, plus 3 refits | The unit a pretrained model is compared in |
| Seconds | measured, on a laptop CPU | The cost the adoption memo reports |

> **A budget is part of the result.** "AUC 0.78 after 103 fits" can be compared; "AUC 0.78" cannot.

#### [day1#search-space] Six parameters, each with a range that a practitioner would defend

*What the search draws from*

```python
# file: bustan.sample_config
{"learning_rate":     log-uniform 0.02 to 0.2,
 "num_leaves":        7, 15, 31 or 63,
 "min_child_samples": 10, 20, 50 or 100,
 "n_estimators":      100, 200 or 400,
 "colsample_bytree":  0.6, 0.8 or 1.0,
 "reg_lambda":        0, 1 or 5}
```

The best of 20 in the course run: 7 leaves, learning rate 0.021, 400 trees, 60% of features per tree. Small trees and many of them.

#### [day1#customers-repeat] A customer seen in training is recognised, not predicted

*Why the folds must be grouped*

- A customer's **distance and segment never change**: together they fingerprint the customer.
- Some customers are late more often for reasons in no column: their street, their receiving hours.
- Random folds put the same customer on both sides; the model learns the fingerprint.
- Bustan predicts for **new and changing customers**, so the folds keep each customer on one side.

- **In the sample.** 5,000 orders from fewer than 1,500 customers: most customers appear more than once.

#### [day1#random-folds-flatter] Random K-fold flattered the same model by 0.02 AUC

*Lab 1.1, the course run*

| Item | Value |
| --- | --- |
| Random 5-fold | 0.799 |
| Grouped 5-fold | 0.779 |
| Held-out customers | 0.785 |

Same configuration, same 5,000 orders. The held-out set is 14,289 orders of customers the model never saw.

#### [day1#top-three-ensemble] Average the top three configurations to make the baseline stable, not lucky

*The last step of the baseline*

- **Why.** The single best of 20 is partly the best by chance. Three good ones disagree in different places.
- **How.** Refit each on all training orders; average the **probabilities**.

**0.784** held-out AUC of the three-model average; the single best scored 0.785

> Here the average did not raise the score. It is kept because the catalog asks for a tuned, ensembled baseline: a pretrained model must beat the strongest version.

#### [day1#checkpoint-1] Which estimate goes in the memo?

*Checkpoint*

**Question:** Sarab's baseline scores 0.80 AUC under random 5-fold and 0.78 under 5-fold grouped by customer. Bustan will use the model on next quarter's orders, many from new customers. Which number goes in the memo?

- A. 0.80, because it is the model's best result
- B. 0.78, because the folds keep each customer on one side
- C. 0.79, the average of the two schemes
- D. Either, as long as the memo names the scheme

#### [day1#lab-1-1] A tuned baseline under a budget

Lab 1.1 · 25 min

Tune a boosted-tree baseline under a fixed budget with grouped cross-validation: twenty random configurations on 5,000 orders, grouped by customer, against random K-fold, then a three-model ensemble (AI-off).

[D1_1_tune_boosted_tree.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D1_1_tune_boosted_tree.ipynb)

- [ ] Load the orders and fix the budget (3 min)
- [ ] Grouped folds by customer (5 min)
- [ ] Random search under the budget (8 min)
- [ ] Grouped against random K-fold (5 min)
- [ ] Ensemble the top three (4 min)
- [ ] ✱ Stretch: Spend the same budget on a finer grid of two parameters

### Session 2: In-context learning on tables

#### [day1#s2] In-context learning on tables

*Session 2*

- Describe what a tabular foundation model learns in pretraining and what it does at prediction
- Run a pretrained tabular model on the same folds as the baseline
- Compare accuracy and run time with the tuning time included

#### [day1#training-rows-as-context] It reads your training rows as context and predicts in one pass, with no training on them

*What a tabular foundation model does*

1. **Pretraining** once, by the vendor, on synthetic tables
2. **Context** your labelled training rows
3. **Query** your test rows
4. **Output** probabilities, one forward pass

> **No parameter updates.** "Fit" stores the rows; the work happens at prediction.

#### [day1#cost-moves-to-predict] The cost moves from fitting to predicting

*Where the seconds go*

|  | Boosted trees | In-context model |
| --- | --- | --- |
| Fit | builds trees | stores rows |
| Tuning | many fits | usually none |
| Predict | walks trees: cheap | attends to training rows: grows with both sizes |
| Hardware | CPU | GPU for large tables |

- **For Bustan.** Scoring 2,000 orders every night is cheap either way. Scoring 50,000 with 50,000 rows of context is not.

#### [day1#two-families] Two families of tabular foundation models, with different licences

*The models in the tool slot*

- **TabICL (TabICLv2).** Classification and regression. Code and weights under BSD-3-Clause, a permissive licence.
- **TabPFN family (TabPFN-2.5, 2.6).** Weights under Prior Labs' non-commercial licences: testing and research, not production, without a commercial licence.

> **Names and licences change.** Checked on the official pages in November 2026. Check again before a cohort.

#### [day1#same-folds-same-metric] Score both models on the same folds and pair the differences

*The rule of Lab 1.2*

```python
# file: Lab 1.2
folds = U.grouped_folds(data.customer, 5)
gbm = fold_scores(lambda: U.lgbm(**best_cfg),
                  X, y, folds)
pre = fold_scores(make_pretrained, X, y, folds)
diff = pre.auc - gbm.auc        # fold by fold
U.mean_ci(diff)
```

- **Paired.** A hard fold is hard for both models; the difference cancels it.
- **Unpaired.** Each model on its own split adds split noise to the comparison.

#### [day1#comparison-with-tuning] At 2,000 orders the two cannot be told apart, and one of them needed 40 seconds of tuning

*Lab 1.2, the course run (stand-in)*

| Answered by | AUC (5 folds) | Fit s | Predict s | Tuning s |
| --- | --- | --- | --- | --- |
| Tuned LightGBM | 0.779 ± 0.027 | 0.39 | 0.008 | 40 |
| Stand-in | 0.771 ± 0.035 | 0.23 | 0.073 | 0 |

Paired difference, stand-in minus baseline: -0.008, 95% interval -0.022 to +0.006.

#### [day1#preprocessing-moves] A pretrained model brings its own preprocessing: know what it does to your columns

*What changed besides the model*

- **Scaling.** Rescaled inside the model. Trees never needed it.
- **Categories.** The segment is stored as a code 0 to 3. A careless pipeline feeds it as a number.
- **Ensembling.** Several passes over transformed copies of the table (TabICL's `n_estimators`).

> **Fairness cuts both ways.** The pretrained model ensembles by default; that is why the baseline is ensembled too.

#### [day1#checkpoint-2] What makes the run-time comparison fair?

*Checkpoint*

**Question:** The pretrained model took 0.3 seconds per fold and the baseline 0.4. Rafif concludes the pretrained model is cheaper. What is missing from his comparison?

- A. The type of processor each model ran on
- B. Nothing: fit time is the cost that matters
- C. The 103 fits the baseline's tuning cost before that fit
- D. The AUC of each model on the same folds

#### [day1#lab-1-2] A pretrained model on the same folds

Lab 1.2 · 25 min

Run a pretrained tabular model on the same splits; compare accuracy and run time: the same grouped folds of 2,000 orders, the model's own preprocessing, and a table that counts the baseline's tuning time.

[D1_2_run_pretrained_tabular.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D1_2_run_pretrained_tabular.ipynb)

- [ ] Who answers: the model or the stand-in (3 min)
- [ ] Same folds, same metric (6 min)
- [ ] Fit and predict, timed (7 min)
- [ ] The comparison table with tuning time (6 min)
- [ ] What changed in the preprocessing (3 min)
- [ ] ✱ Stretch: Give the pretrained model the raw categories and compare

### Session 3: Learning curves decide

#### [day1#s3] Learning curves decide

*Session 3*

- Build learning curves from repeated subsamples at six training sizes
- Put an interval on each point and read where the curves cross
- State where each approach wins and where the evidence is too thin to say

#### [day1#one-score-one-point] A score at one training size is one point on a curve

*Why one comparison is not enough*

- Bustan has **500 orders** for a new segment and **50,000** for its oldest depots.
- Pretrained models are often strongest on small tables; tuned trees catch up as rows grow.
- Where the curves cross, if they cross, is the decision.

- **Six sizes.** 500, 1,000, 2,000, 5,000, 10,000 orders in the lab; 50,000 supplied.

#### [day1#repeats-make-intervals] Repeat every point on a fresh subsample to see how far it moves

*The design of Lab 1.3*

1. **Draw** *n* orders from the pool
2. **Tune** the baseline: 8 configurations, grouped folds
3. **Score** both on the same held-out customers
4. **Repeat** with a new seed

> **The interval is over samples.** It answers "what if Bustan had drawn other orders of this size?"

#### [day1#curves-overlap] On Bustan's late deliveries the two curves overlap at every size

*Lab 1.3, the course run (stand-in)*

| Training orders | Tuned LightGBM, AUC (95% interval) | Stand-in, AUC (95% interval) |
| --- | --- | --- |
| 500 | 0.740 (0.704 to 0.775) | 0.750 (0.725 to 0.775) |
| 2,000 | 0.768 (0.733 to 0.804) | 0.765 (0.754 to 0.775) |
| 10,000 | 0.777 (0.747 to 0.807) | 0.774 (0.770 to 0.778) |
| 50,000 (supplied) | 0.775 (0.764 to 0.786) | 0.777 (0.777 to 0.778) |

The paired difference at every size has an interval that contains zero: no size where one approach wins.

#### [day1#reading-a-crossover] A crossover counts only where the interval of the difference leaves zero

*How to read a learning curve*

- **Pretrained wins.** The whole interval of (pretrained minus baseline) above zero.
- **Baseline wins.** The whole interval below zero.
- **Cannot tell.** The interval contains zero. Say so, and say what more evidence would decide it.

> **Overlapping error bars are not the test.** Two bars can overlap while the paired difference is clear, because the repeats are paired.

#### [day1#beyond-the-last-point] Do not extend a curve past its last measured size

*What the curve does not say*

- The supplied 50,000-order point exists because the curve could not be guessed from 10,000.
- Run time grows differently: in the course run, 24 seconds for the baseline at 50,000 (tuning included) and 21 for the stand-in.
- A real in-context model at 50,000 rows needs a GPU; that is a cost line, not a footnote.

#### [day1#checkpoint-3] What does this point say?

*Checkpoint*

**Question:** At 500 orders, three repeats give the pretrained model a mean AUC 0.011 above the tuned baseline. The 95% interval of the paired difference runs from -0.031 to +0.052. What goes in the memo?

- A. No difference detectable at 500 orders with three repeats
- B. The pretrained model wins on small data
- C. The two models are equally accurate at 500 orders
- D. The comparison is invalid and should be dropped

#### [day1#lab-1-3] Learning curves with intervals

Lab 1.3 · 25 min

Plot both models across six training sizes, using supplied scores for the largest: repeated subsamples at 500 to 10,000 orders, intervals per point, and the supplied 50,000-order scores.

[D1_3_plot_both_across.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D1_3_plot_both_across.ipynb)

- [ ] The sizes and the repeats (4 min)
- [ ] Score both models at each size (9 min)
- [ ] Intervals per point (5 min)
- [ ] Add the supplied largest size and plot (4 min)
- [ ] Where each approach wins (3 min)
- [ ] ✱ Stretch: Add the run time per size as a second panel

### Session 4: Reading a leaderboard claim

#### [day1#s4] Reading a leaderboard claim

*Session 4*

- Find unequal tuning budgets in a benchmark table
- Remove datasets the pretrained model may have seen or been selected on
- Recount wins as a paired difference with an interval

#### [day1#vendor-claim] "Beats gradient boosting on 35 of 36 benchmark datasets"

*Rafif's email*

**Rafif:** Look at this slide from the vendor. 35 out of 36! Should we just switch?

**Sarab:** Send me the table behind it, not the slide.

- **Three questions for any claim.** Was the baseline tuned as much? Was the model scored on data it was built or selected on? How big is each win, with an interval?

#### [day1#unequal-budgets] The claim's boosted trees got zero tuning trials; the pretrained model ensembled eight passes

*The first column to read*

- **35 / 36** wins as claimed: defaults against an ensemble
- **23 / 36** wins against trees tuned with 100 trials
- **11 / 23** wins once overlapping datasets are removed

#### [day1#pretraining-overlap] A model scored on datasets it was selected on is grading its own homework

*The second column to read*

- **Pretraining overlap**: the dataset, or a copy, was in the training data.
- **Selection overlap**: the dataset was used to choose the model's settings or checkpoint.
- Either inflates the score on exactly that dataset.

- **In the table.** 13 of 36 datasets carry the flag. Drop them before counting.

> Pretraining on synthetic tables lowers the risk of the first kind; public validation suites keep the second alive.

#### [day1#paired-difference-slices] Without overlap the average advantage is zero; it splits by table size

*The third question*

| Clean datasets | Count | Pretrained minus tuned, AUC | 95% interval |
| --- | --- | --- | --- |
| All | 23 | -0.0005 | -0.005 to +0.004 |
| Under 3,000 rows | 10 | +0.009 | +0.005 to +0.013 |
| 3,000 rows or more | 13 | -0.008 | -0.011 to -0.005 |

A bootstrap over datasets. Counting wins hid both effects.

#### [day1#own-curve-beats-claim] Someone else's benchmark sets a hypothesis; your own curve tests it

*The reply*

Prompt for Reply to Rafif:

```text
The claim compared an eight-member pretrained ensemble with untuned trees. At equal tuning and without the 13 overlapping datasets, the advantage is zero on average: positive under 3,000 rows, negative above. On our late-delivery data the two approaches cannot be told apart at any size from 500 to 50,000 orders.
```

#### [day1#checkpoint-4] Which fact voids the claim first?

*Checkpoint*

**Question:** A benchmark table shows a pretrained model winning 30 of 32 datasets. Which single fact, if true, most undermines the claim?

- A. The pretrained model was slower on every dataset
- B. Most of the datasets are publicly available
- C. Half of the datasets have fewer than 3,000 rows
- D. The boosted trees were run at default settings

#### [day1#lab-1-4] Check a leaderboard claim

Lab 1.4 · 25 min

Check a claim for unequal tuning budgets and pretraining overlap: recount a vendor benchmark table at equal budgets, drop overlapping datasets and put an interval on the paired difference.

[D1_4_check_claim_unequal.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D1_4_check_claim_unequal.ipynb)

- [ ] Read the claim and its table (4 min)
- [ ] Equal budgets (6 min)
- [ ] Drop overlap (5 min)
- [ ] Paired difference with an interval (6 min)
- [ ] Set it beside your own curve (4 min)
- [ ] ✱ Stretch: Count how many datasets the claim would need to hold at equal budget

#### [day1#wrap-up] A fair comparison is designed before it is run

*Day 1 · Three things to keep*

- **Budget and folds.** Write the tuning budget down; keep each customer on one side.
- **Curves, not points.** Compare across sizes, with intervals of the paired difference.
- **Claims are hypotheses.** Equal budgets, no overlap, an interval; then your own data.

#### [day1#next] Forecasting with pretrained models

*Tomorrow · Day 2*

Zero-shot forecasts on 400 series against supplied backtests, the groups where a benchmark wins, Ramadan and Eid as covariates, and stacking.

- Recall DS-304: the benchmark floor, rolling origins, MASE.
- Colab users: Lab 2.1 installs a pretrained forecaster; start it early.

## Day 2: Forecasting with pretrained models

Zero-shot forecasts on 400 series, by group, with holidays and stacking

#### [day2#title] Forecasting with pretrained models

*SDAIA Academy · Advanced Modelling for Structured Data*

Day 2: Zero-shot forecasts on 400 series, by group, with holidays and stacking

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-404

#### [day2#agenda] Forecasting with pretrained models

*Today*

- **Session 1 · Zero-shot against a benchmark floor**: A zero-shot forecaster is one more method in the same rolling-origin backtest, scored on the same windows
- **Session 2 · Where zero-shot loses**: An average over 400 series hides the groups where a benchmark wins
- **Session 3 · Holidays that move**: Ramadan comes about eleven days earlier each Gregorian year, so a yearly lag misses it and a covariate finds it
- **Session 4 · Ensembles worth their cost**: A stack learned on earlier windows must beat the best single model on later windows by more than its interval

#### [day2#review-day1] What would change Sarab's verdict?

*Review of Day 1*

**Question:** Yesterday the learning curves of the tuned baseline and the pretrained model overlapped at every size, with paired intervals around zero. Which new evidence would justify writing "the pretrained model wins below 1,000 orders"?

- A. A vendor table with 35 wins out of 36
- B. A higher mean AUC at 500 orders in one more run
- C. More repeats at 500 and 1,000 orders whose paired interval stays above zero
- D. Rerunning the curves with random K-fold instead of held-out customers

#### [day2#recap-backtest] A forecaster counts only if it beats the floor in a rolling-origin backtest

*Recap from DS-304, not retaught*

1. **Origin** cut the history at a date
2. **Forecast** 28 days ahead from what was known
3. **Score** against what happened
4. **Roll** to the next origin

> DS-304 built the floor, the smoothing and ARIMA models, Hijri regressors and the simple average. Today they are **supplied**: six origins, 400 series, three benchmarks.

### Session 1: Zero-shot against a benchmark floor

#### [day2#s1] Zero-shot against a benchmark floor

*Session 1*

- Run a zero-shot forecaster on 400 series at four origins
- Score scaled point error and quantile loss on the same windows as supplied benchmarks
- Report each method against the seasonal naive floor

#### [day2#zero-shot-is-one-more-method] A zero-shot forecaster needs no training on your series, but it is scored like any other method

*What changes with a pretrained forecaster*

- **Input**: each series' history up to the origin. No product, no depot, no features unless you pass covariates.
- **Output**: quantiles for the next 28 days.
- **No fit** on Bustan's data, so no tuning budget, and nothing to retrain each month.
- **Same windows** as the benchmarks, or the comparison means nothing.

- **Chronos-2.** A pretrained model for univariate, multivariate and covariate-informed forecasting, trained on synthetic data. Weights under Apache-2.0.

#### [day2#scaled-scores] Divide every error by the series' own scale, or the staples decide alone

*Two numbers per series and origin*

| Score | What it measures | Read |
| --- | --- | --- |
| **MASE** | mean absolute error of the median, divided by the in-sample error of last week's copy | below 1 beats the naive in-sample |
| **Scaled quantile loss** | pinball loss over the 0.1, 0.5 and 0.9 quantiles, same scale | rewards sharp bands that still hold |
| **80% coverage** | share of days inside the 0.1 to 0.9 band | should be near 80% |

#### [day2#pinball-loss] The pinball loss charges a quantile forecast unequally for being low and being high

*The loss behind every quantile*

```python
# file: bustan.pinball
def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)

# tau = 0.9: a forecast 10 below costs 9,
# a forecast 10 above costs 1
```

- **Tomorrow's decision.** The same asymmetry is the newsvendor's: when a shortage costs nine times a leftover, order the 0.9 quantile. Day 3 starts here.

#### [day2#zero-shot-beats-floor] The zero-shot forecaster beats the floor and smoothing, not the tuned LightGBM

*Lab 2.1, the course run (stand-in), four origins, 1,504 series-origins*

| Method | MASE | Scaled quantile loss | Loss against the floor | 80% coverage |
| --- | --- | --- | --- | --- |
| Seasonal naive (floor) | 1.141 | 0.811 | 1.000 | 75% |
| Smoothing | 0.973 | 0.676 | 0.833 | 71% |
| Zero-shot (stand-in) | 0.883 | 0.610 | 0.751 | 74% |
| LightGBM, tuned | 0.722 | 0.491 | 0.605 | 81% |

#### [day2#same-windows] Same origins, same horizon, same series, same scale

*What makes the comparison fair*

- **Fair.** The zero-shot model is run at the four origins of the supplied backtests and scored with the same function.
- **Unfair.** A vendor's accuracy on its benchmark, set beside Bustan's backtest. Different series, different windows, different scale.

> **Check it in code.** Lab 2.1 asserts that your scores match the helper's before printing a table.

#### [day2#checkpoint-1] Is the zero-shot forecaster useful here?

*Checkpoint*

**Question:** The zero-shot forecaster scores 0.75 of the floor's quantile loss; the tuned LightGBM scores 0.61. Ahlam asks whether the pretrained model is worth keeping. What is the best answer today?

- A. Not as a replacement; test where it wins and what it adds to an ensemble first
- B. No: it loses to LightGBM, so it has no use
- C. Yes: it needs no training, so it should replace LightGBM
- D. Yes: it beats the floor, which is all a forecaster must do

#### [day2#lab-2-1] Zero-shot against the supplied backtests

Lab 2.1 · 25 min

Score zero-shot forecasts for 400 series against supplied benchmark backtests: the same four origins and 28-day horizon, scaled error, quantile loss and 80% coverage against a seasonal naive, a smoothing model and a tuned LightGBM.

[D2_1_score_zero_shot.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D2_1_score_zero_shot.ipynb)

- [ ] Who answers, and the supplied backtests (3 min)
- [ ] Zero-shot at four origins (7 min)
- [ ] Scaled error and quantile loss (8 min)
- [ ] Against the floor (4 min)
- [ ] Coverage of the 80% band (3 min)
- [ ] ✱ Stretch: Shorten the context to 120 days and score again

### Session 2: Where zero-shot loses

#### [day2#s2] Where zero-shot loses

*Session 2*

- Classify series as smooth, intermittent or short from their own history
- Compare methods by series group, point and quantile
- Turn a group table into a routing rule

#### [day2#averages-hide-groups] An average over 400 series hides the groups where a method fails

*Why one number is not enough*

- **Regular.** 160 series: staples and dairy, sold most days.
- **Intermittent.** 100 series: specialty imports and catering items, many zero days.
- **Ramadan items.** 80 series: dates and juices, from the product master.
- **Short.** 60 series: products launched in the last six months.

#### [day2#classify-from-history] Two numbers describe a demand pattern: how often it sells, and how much the sizes vary

*Classifying from the series itself*

- **ADI**: average days between non-zero days over the last year.
- **CV squared**: variability of the non-zero sizes.
- In the lab: **short** under 180 days of history; **intermittent** when ADI is 1.32 or more; the rest regular.

- **A class has a date.** A product launched in December is short in October and regular by May. Classify at the origin you forecast from.

#### [day2#scores-by-group] The tuned LightGBM wins every group; the zero-shot gap is smallest on intermittent items

*Lab 2.2, the course run (stand-in), scaled quantile loss*

| Group | Seasonal naive | Smoothing | Zero-shot | LightGBM | Zero-shot gap |
| --- | --- | --- | --- | --- | --- |
| Intermittent | 0.792 | 0.662 | 0.501 | 0.469 | +0.032 |
| Regular | 0.715 | 0.559 | 0.551 | 0.440 | +0.112 |
| Short | 0.958 | 0.848 | 0.839 | 0.653 | +0.186 |
| Ramadan items | 0.962 | 0.851 | 0.760 | 0.549 | +0.211 |

The gap is the paired difference, zero-shot minus LightGBM; every interval excludes zero. Against smoothing, the zero-shot model ties on short series.

#### [day2#short-series-borrow] A short series borrows strength from other series, or it has nothing to learn from

*Recap from DS-304, applied*

- **Global model.** The LightGBM learns from all 400 series at once, so a new product starts with its category's pattern.
- **Zero-shot, one series.** Sees only the product's own few weeks; the stand-in falls back to its last eight weeks.
- **In context, many series.** Models that read a group of related series together can borrow without training.

#### [day2#routing-rule] Choose the method per group on earlier windows, then score it on later ones

*From a group table to a rule*

| Group | Chosen on 2025-10-05 and 2025-11-16 | Why |
| --- | --- | --- |
| Regular, intermittent, Ramadan items | LightGBM, tuned | lowest loss on the earlier windows |
| Short | zero-shot | no short series existed then: a method that needs no history |

> **A rule picked on the scored windows scores itself.** It will look better than it will be next quarter.

#### [day2#checkpoint-2] What do you do with this group?

*Checkpoint*

**Question:** For 60 short series the zero-shot forecaster and smoothing differ by -0.009 in scaled quantile loss, with a 95% interval from -0.064 to +0.050. Both trail LightGBM by about 0.19. What do you recommend for new products?

- A. Switch new products to the zero-shot model, which scored lower
- B. Keep smoothing because the two are proven equal
- C. Treat short series as unforecastable and order by hand
- D. Use the tuned LightGBM, which wins the group, and retest the pretrained model later

#### [day2#lab-2-2] Where the benchmark wins

Lab 2.2 · 25 min

Find the series groups, such as intermittent items, where the benchmark wins: classify by zero share and variability, compare methods per group and write a routing rule.

[D2_2_find_series_groups.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D2_2_find_series_groups.ipynb)

- [ ] Classify the series (7 min)
- [ ] Scores by group (7 min)
- [ ] Where does the benchmark win? (6 min)
- [ ] A routing rule (5 min)
- [ ] ✱ Stretch: Score the routed forecast as a fifth method

### Session 3: Holidays that move

#### [day2#s3] Holidays that move

*Session 3*

- Show why a yearly lag must respect both the week and the Hijri year
- Supply Ramadan and Eid as known future covariates
- Measure error in holiday weeks separately from ordinary weeks

#### [day2#two-cycles] Demand follows the week and the Hijri year at once, and they do not line up

*Why holidays break yearly lags*

- **364 days.** 52 weeks: same weekday, but Ramadan has moved about eleven days.
- **354 days.** One Hijri year: right Ramadan day, wrong weekday.
- **357 days.** 51 weeks: same weekday, within three days of the Hijri date.

Ramadan 1446 began on 1 March 2025; Ramadan 1447 on 18 February 2026 (Umm al-Qura).

#### [day2#three-lags-result] From pre-Ramadan to Eid al-Fitr, the 51-week lag beats both yearly lags

*Lab 2.3, the course run*

| Item | Value |
| --- | --- |
| Lag 354 days | 0.615 |
| Lag 364 days | 0.574 |
| Lag 357 days | 0.487 |

Mean absolute error relative to the window's mean, 80 Ramadan items, 4 February to 23 March 2026.

#### [day2#known-future-covariates] The calendar is a known future covariate; next week's sales are not

*What may go into a forecast*

| Input | Known at the origin? | Use |
| --- | --- | --- |
| Pre-Ramadan, Ramadan, pre-Eid, Eid days | yes, published in advance | future covariate |
| A planned promotion | yes, if decided before the origin | future covariate |
| Temperature | only as a forecast | with care: pass the forecast, not the actual |
| Next week's sales of a related product | no | leakage |

#### [day2#covariates-in-code] The covariates for the past and for the next 28 days go in the same call

*Passing the calendar*

```python
# file: bustan.zero_shot (Colab path)
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
pred = pipe.predict_df(
    context_df,              # id, timestamp, target, ramadan, pre_eid, ...
    future_df=future_df,     # id, timestamp, ramadan, pre_eid, ... (next 28 days)
    prediction_length=28,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id", timestamp_column="timestamp", target="target")
```

#### [day2#holiday-split] Covariates cut the zero-shot holiday-day loss by 10% and did not hurt ordinary days

*Lab 2.3, the course run (stand-in), scaled quantile loss*

| Method | Holiday days | Ordinary days |
| --- | --- | --- |
| Smoothing | 0.808 | 0.608 |
| Zero-shot | 0.744 | 0.517 |
| Zero-shot + covariates | 0.666 | 0.482 |
| LightGBM, tuned (had the calendar) | 0.471 | 0.473 |

On the 80 Ramadan items alone: 0.997 without covariates, 0.727 with them.

#### [day2#checkpoint-3] Which lag for a Ramadan item?

*Checkpoint*

**Question:** A planner forecasts dates (the fruit) for next Ramadan by copying last year. Bustan's demand is low on Fridays. Which single lag gives the smallest error over Ramadan and Eid?

- A. 364 days, one Gregorian year in whole weeks
- B. 357 days, 51 whole weeks
- C. 354 days, one Hijri year
- D. 365 days, one calendar year

#### [day2#lab-2-3] Ramadan and Eid as covariates

Lab 2.3 · 25 min

Add Ramadan and Eid covariates and measure holiday-week error: the 364-, 357- and 354-day lags, future covariates for the zero-shot model, and error split by holiday and ordinary weeks.

[D2_3_add_ramadan_eid.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D2_3_add_ramadan_eid.ipynb)

- [ ] Three lags: 364, 357 and 354 days (5 min)
- [ ] Build the future covariates (6 min)
- [ ] Zero-shot with covariates (6 min)
- [ ] Holiday weeks against ordinary weeks (8 min)
- [ ] ✱ Stretch: Drop the pre-Eid flag and see which weeks suffer

### Session 4: Ensembles worth their cost

#### [day2#s4] Ensembles worth their cost

*Session 4*

- Build a stacking set from rolling-origin forecasts without leakage
- Fit non-negative stacking weights and test them on later windows
- Report the improvement over the best single model with a bootstrap interval

#### [day2#recap-simple-average] The simple average is hard to beat; learned weights must earn their extra risk

*Recap from DS-304, then one step further*

- **Simple average.** No weights to estimate, so nothing to overfit. Works when the members are comparably good.
- **Stacking.** Weights learned from earlier forecasts. Wins when the members differ in quality, if the weights are estimated honestly.

#### [day2#rolling-origin-stacking] Weights for an origin come only from windows that ended before it

*How the weights are learned*

| Scored origin | Weights fitted on |
| --- | --- |
| 2026-01-04 | 2025-10-05, 2025-11-16 |
| 2026-02-15 | the three before it |
| 2026-03-15 | the four before it |
| 2026-05-10 | the five before it |

> **The rolling-origin forecasts are the out-of-fold predictions of time series.** Each was made without its own answers.

#### [day2#fit-the-judged-loss] Fit the weights on the loss you will be judged on

*Lab 2.4, the course run (stand-in)*

| Fitted on | LightGBM | Smoothing | Zero-shot + covariates | Floor |
| --- | --- | --- | --- | --- |
| Scaled quantile loss | 0.81 to 0.95 | 0 | 0.05 to 0.19 | 0 |
| Least squares on medians | 0.58 to 0.61 | 0.18 to 0.39 | 0 to 0.24 | 0 |

Ranges over the four scored origins. Least squares chases the large Ramadan spikes and gives smoothing weight it does not deserve on quantile loss.

#### [day2#stack-gain-interval] The stack's gain is real and tiny: 0.4%, for running four models

*Lab 2.4, the course run (stand-in), against the best single model*

| Item | Value |
| --- | --- |
| Stack, judged loss | +0.4% (+0.0012 to +0.0028) |
| Least-squares stack | -7.5% (-0.041 to -0.033) |
| Simple average | -18% (-0.094 to -0.083) |

Gain in scaled quantile loss over the best single model (always the tuned LightGBM), bootstrap over series.

#### [day2#ensemble-cost-line] An ensemble is adopted when its gain beats its running cost

*The decision*

- **Gain.** With an interval, against the best single model chosen fairly.
- **Cost.** Every member fitted, run and monitored every cycle; one more licence to check.
- **Decision.** Here: keep LightGBM alone, and keep the zero-shot model on watch for holiday windows.

#### [day2#checkpoint-4] Was the stack fitted honestly?

*Checkpoint*

**Question:** An analyst fits stacking weights on all six origins and reports that the stack beats the best single model on origins three to six. What is wrong?

- A. Four members are too many for six origins
- B. The weights were fitted on quantile loss
- C. The weights saw the windows they are scored on
- D. The best single model was the tuned LightGBM

#### [day2#lab-2-4] Stacking on rolling-origin windows

Lab 2.4 · 25 min

Stack forecasters on rolling-origin windows and test the improvement against its interval: four fitted candidates, weights learned on earlier origins, a series bootstrap for the gain over the best single model.

[D2_4_stack_forecasters_rolling.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D2_4_stack_forecasters_rolling.ipynb)

- [ ] The four candidates and the windows (4 min)
- [ ] Weights from earlier windows only (8 min)
- [ ] Best single, average and stack on later windows (6 min)
- [ ] The gain and its interval (7 min)
- [ ] ✱ Stretch: Learn one weight set per series group

#### [day2#wrap-up] Score a pretrained forecaster like any other method, by group and by holiday

*Day 2 · Three things to keep*

- **Same windows.** The supplied backtest's origins, horizon and scale.
- **Groups and holidays.** Averages hide where a method fails; the calendar is a covariate.
- **Ensembles pay or go.** Weights from earlier windows, on the judged loss, with an interval.

#### [day2#next] From forecast to order

*Tomorrow · Day 3*

Newsvendor quantities, an order and truck programme for one depot week, forecasters ranked by regret, and the plan under simulated demand.

- Revisit the pre-work on newsvendor logic and linear programmes.
- Run the setup check's small optimisation once more: it proves the solver works.

## Day 3: From forecast to order

Quantiles into orders, trucks and regret

#### [day3#title] From forecast to order

*SDAIA Academy · Advanced Modelling for Structured Data*

Day 3: Quantiles into orders, trucks and regret

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-404

#### [day3#agenda] From forecast to order

*Today*

- **Session 1 · Asymmetric costs**: When a shortage costs more than a leftover, order a high quantile of demand, not its mean
- **Session 2 · Optimising under capacity**: A rule per product ignores shared trucks and space: an integer programme decides them together
- **Session 3 · Scoring by regret**: Judge a forecaster by the cost of the decisions it causes, not only by its error
- **Session 4 · Uncertainty in the plan**: A plan that works on the forecast can still run short: simulate demand and report the risk

#### [day3#review-day2] Is a 0.4% gain worth four models?

*Review of Day 2*

**Question:** Yesterday's stack beat the tuned LightGBM by 0.4% in scaled quantile loss, with a 95% interval from +0.0012 to +0.0028. What is still missing before Ahlam can decide?

- A. Nothing: the interval excludes zero, so adopt the stack
- B. Six more origins to narrow the interval
- C. What 0.4% less loss is worth in riyals, against the cost of running four models
- D. A comparison with the seasonal naive floor

### Session 1: Asymmetric costs

#### [day3#s1] Asymmetric costs

*Session 1*

- Compute underage and overage costs from a cost table
- Turn a critical ratio into an order quantity from quantile forecasts
- Measure the cost of ordering the median instead

#### [day3#two-costs] A case short and a case left over do not cost the same

*What a wrong order costs*

- **Underage: one case short.** The lost margin, price minus cost, plus goodwill worth half the margin: the customer may order elsewhere next week.
- **Overage: one case left over.** Holding at 0.4% of the cost per day for a week; for products that keep 30 days or less, 60% of the cost lost to spoilage.

> Bustan's cost table is synthetic: the rates are the course's, set to look like a food wholesaler's, not quoted from one.

#### [day3#critical-ratio] Order up to the demand quantile at underage / (underage + overage)

*The newsvendor result*

1. **Costs** underage, overage per case
2. **Critical ratio** u / (u + o)
3. **Quantile** of the week's demand
4. **Order** rounded up to cases

> **Takeaway.** The mean is the right order only when a shortage and a leftover cost the same.

#### [day3#ratios-by-category] Same forecaster, opposite decisions: dry goods above the median, dairy below it

*Lab 3.1, the course run, 20 products at the Riyadh depot*

| Category | Critical ratio | Order |
| --- | --- | --- |
| Rice, flour, sugar | 0.82 to 0.92 | well above the median |
| Beverages | 0.83 to 0.92 | well above the median |
| Dates | 0.83 to 0.91 | well above the median |
| Chilled dairy (21 days) | 0.17 to 0.33 | below the median |

#### [day3#weekly-from-daily] Add the daily medians; add the daily variances, not the spreads

*The forecaster gives days; Imad orders weeks*

```python
# file: labs/_src/D3_1_compute_newsvendor_quantities.py
mu = q[:, 1].sum(axis=1)                                    # sum of daily medians
sd = np.sqrt((((q[:, 2] - q[:, 0]) / 2.563) ** 2).sum(axis=1))  # 0.1 to 0.9 spans 2.563 sd
order = np.ceil(np.maximum(mu + sd * stats.norm.ppf(cr), 0))
```

> **Two assumptions.** Days are independent and the bands are right. Session 4 tests both on Bustan's own errors.

#### [day3#median-costs-more] Ordering the median costs a third more than ordering the critical quantile

*Lab 3.1, the course run, expected cost of one week (20,000 simulated weeks)*

| Item | Value |
| --- | --- |
| Order the median | SAR 10,592 |
| Newsvendor order | SAR 7,127 |

Saving SAR 3,465 a week (33%) for 20 products at one depot. Chilled dairy is SAR 4,984 of the newsvendor's cost.

#### [day3#checkpoint-1] Which quantile do you order?

*Checkpoint*

**Question:** A chilled yoghurt line: a case short costs SAR 12 in margin and goodwill; a case left over costs SAR 36, mostly spoilage. Which quantile of next week's demand should Imad order?

- A. The 0.75 quantile
- B. The 0.33 quantile
- C. The median
- D. The 0.25 quantile

#### [day3#lab-3-1] Newsvendor quantities

Lab 3.1 · 25 min

Compute newsvendor quantities from quantile forecasts for 20 products: underage and overage from the cost table, the critical ratio, the matching quantile of weekly demand and the cost of ordering the median.

[D3_1_compute_newsvendor_quantities.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D3_1_compute_newsvendor_quantities.ipynb)

- [ ] Underage and overage from the cost table (5 min)
- [ ] The critical ratio (4 min)
- [ ] Weekly quantiles from daily ones (6 min)
- [ ] Order quantities for 20 products (5 min)
- [ ] What ordering the median costs (5 min)
- [ ] ✱ Stretch: Change the shortage cost of dates and watch the quantity move

### Session 2: Optimising under capacity

#### [day3#s2] Optimising under capacity

*Session 2*

- Write an order and truck plan as variables, constraints and an objective
- Solve it with an open solver and read the solution
- Find which constraint binds and what relaxing it is worth

#### [day3#shared-limits] The 20 orders share trucks, two reefers and 200 cubic metres of depot space

*Why one rule per product is not enough*

- **Trucks.** 60 m3 and 20 tonnes, SAR 2,400 a trip, one supplier per truck.
- **Reefers.** Dairy only: 40 m3 and 14 tonnes, SAR 3,200 a trip, two a week.
- **Depot space.** 200 m3 received a week at Riyadh. The newsvendor orders need 218.8.

#### [day3#integer-programme] Variables, an objective and constraints, solved together by HiGHS

*From rule to programme*

```python
# file: labs/_src/D3_2_solve_order_truck.py
q[i]      = prob.add_variable(f"q_{p}", lowBound=0, cat="Integer")   # cases
trucks[s] = prob.add_variable(f"trucks_{s}", lowBound=0, cat="Integer")
prob += trip_cost * trucks + underage * short + overage * over        # objective
prob += q[i] - target[i] == over[i] - short[i]                          # distance from target
prob += lpSum(case_m3 * q[i] for i in s) <= 60 * trucks[s], f"m3_{s}"  # and kg_, reefers
prob += lpSum(case_m3 * q[i]) <= 200, "depot_space"
prob.solve(pulp.HiGHS(msg=False))
```

#### [day3#why-integers] Half a truck costs a whole trip, so trucks are integer variables

*Why whole numbers matter*

- **Continuous.** 1.26 trucks of dry goods at SAR 3,014: a price no driver charges. The plan looks cheaper than it is.
- **Integer.** One truck or two. The solver searches the whole numbers (branch and bound) and proves the best.

#### [day3#plan-result] The programme cuts dry goods to fit one truck, and the depot space binds

*Lab 3.2, the course run, 11 to 17 May 2026*

| Supplier | Trucks | Cut below target |
| --- | --- | --- |
| S1 rice, flour, sugar | 1 | 353 cases: 340 of P003, 12 of P001, 1 of P002 |
| S2 beverages | 2 | none |
| S3 chilled dairy | 2 reefers | none |
| S4 dates | 1 | none |

Objective SAR 21,585. Inbound 200.0 of 200 m3. Ten more cubic metres would save SAR 1,703 that week.

#### [day3#binding-constraints] A constraint binds when it has no slack; it matters when relaxing it lowers the cost

*Reading the solution*

| Constraint | Slack | Relax it by one unit |
| --- | --- | --- |
| depot_space | 0.02 m3 | 10 m3 saves SAR 1,703 |
| reefers | 0 reefers | a third reefer saves nothing: the two have 26.5 m3 free |
| kg_S1 | 1 kg | tight because the cut was sized to fit one truck |

#### [day3#checkpoint-2] Should Imad pay for a third reefer?

*Checkpoint*

**Question:** The plan uses both reefers, so the reefer constraint has zero slack. The two reefers carry 53.5 of their 80 cubic metres. A haulier offers a third reefer for SAR 900 this week. What do you tell Imad?

- A. Accept: the reefer constraint binds, so more reefers help
- B. Decline: re-solving with three reefers would not change the plan, because dairy already fits
- C. Accept: a third reefer frees depot space for dry goods
- D. Accept: it is cheaper than the SAR 1,703 that capacity is worth

#### [day3#lab-3-2] Orders and trucks for one depot week

Lab 3.2 · 25 min

Solve an order and truck-allocation programme for one depot week: 20 products, four suppliers, integer trucks, reefer and depot-space limits, solved with HiGHS through PuLP.

[D3_2_solve_order_truck.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D3_2_solve_order_truck.ipynb)

- [ ] Variables and objective (6 min)
- [ ] Constraints: trucks, reefers, space (7 min)
- [ ] Solve and read the plan (6 min)
- [ ] Which constraint binds (6 min)
- [ ] ✱ Stretch: Add a minimum order per supplier

### Session 3: Scoring by regret

#### [day3#s3] Scoring by regret

*Session 3*

- Compute the realised cost of a plan made from each forecaster
- Compute regret against the plan made with hindsight
- Compare the ranking by regret with the ranking by error

#### [day3#regret-defined] Regret is what the order cost, minus what the best order in hindsight would have cost

*Scoring a forecaster by its decisions*

1. **Forecast** quantiles for the week
2. **Order** the critical quantile
3. **Demand happens** cost of short and left over
4. **Regret** minus the hindsight cost

> **Takeaway.** Without capacity limits the hindsight order is the demand itself, so regret is the realised cost.

#### [day3#error-vs-regret] Error weighs every quantile and product alike; regret weighs them by what they cost

*Why the two scores can disagree*

- **Scaled quantile loss.** The 0.1, 0.5 and 0.9 quantiles of every product, each scaled to its own series.
- **Regret.** The 0.9 quantile of rice and the 0.25 quantile of dairy, in riyals: a dairy leftover costs SAR 40 to 97 a case.

#### [day3#two-rankings] Over twelve weeks the rankings agree; at one origin they flip

*Lab 3.3, the course run, 12 weeks, 20 products, no capacity limit*

| Forecaster | Scaled quantile loss | Regret, 12 weeks | At 2026-05-10: loss | regret |
| --- | --- | --- | --- | --- |
| LightGBM, tuned | 0.421 | SAR 100,672 | 0.384 | SAR 30,761 |
| Zero-shot + covariates (stand-in) | 0.506 | SAR 136,339 | 0.530 | SAR 56,845 |
| Smoothing | 0.598 | SAR 148,737 | 0.565 | SAR 51,832 |

The week's best forecaster differs between the two rankings in 5 of 12 weeks.

#### [day3#regret-interval] Twelve weeks cannot separate LightGBM from the zero-shot forecaster

*Lab 3.3, paired by week*

| Item | Value |
| --- | --- |
| LightGBM minus zero-shot | SAR -2,972 a week (-7,820 to +506) |

Mean difference in weekly regret with a 95% bootstrap interval over the 12 weeks. The interval crosses zero.

#### [day3#checkpoint-3] Why did the rankings flip?

*Checkpoint*

**Question:** At the origin 2026-05-10 the zero-shot forecaster has the lower scaled quantile loss of the two, yet its orders cost SAR 5,000 more than smoothing's over the four weeks. Which explanation fits?

- A. The scaled quantile loss was computed wrongly
- B. Regret depends on demand, not on the forecaster
- C. Its errors fell where they cost most: dairy overage and the Eid al-Adha week
- D. Smoothing is the better forecaster for Bustan

#### [day3#lab-3-3] Regret against error

Lab 3.3 · 25 min

Rank three forecasters by regret and by error: plans from each forecaster's quantiles over twelve weeks, realised cost, regret against hindsight and the two rankings side by side.

[D3_3_rank_three_forecasters.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D3_3_rank_three_forecasters.ipynb)

- [ ] Plans from three forecasters (6 min)
- [ ] Realised cost (5 min)
- [ ] Regret against hindsight (6 min)
- [ ] Two rankings, side by side (8 min)
- [ ] ✱ Stretch: Repeat with a lower shortage cost: does the ranking change?

### Session 4: Uncertainty in the plan

#### [day3#s4] Uncertainty in the plan

*Session 4*

- Simulate weekly demand scenarios with correlated days and products
- Report shortage probability and fill rate of a fixed plan
- Show how independence assumptions understate shortage risk

#### [day3#plan-risk] A plan that is right on average can still run short in the weeks that matter

*What the expected cost hides*

- **Simulate** 2,000 weeks of demand from the forecast and run the fixed plan through them.
- **Report** per product: the share of weeks with a shortage and the fill rate.
- **Check the assumptions** on past errors: are the bands the right width, are days and products independent?

#### [day3#shared-shocks] One shock shared by all products, one per product-week, one per day

*Building correlated scenarios*

```python
# file: labs/_src/D3_4_stress_test_plan.py
a = np.sqrt(rho_prod)                     # all products, same week
b = np.sqrt(max(rho_day - rho_prod, 0))   # one product, all its days
c = np.sqrt(1 - a**2 - b**2)              # the rest, day by day
z = a * N(n, 1, 1) + b * N(n, 20, 1) + c * N(n, 20, 7)
daily = np.maximum(mu + scale * sd * z, 0)
```

#### [day3#measured-errors] The correlations are small; the errors spread about 30% wider than the bands

*Lab 3.4, the course run, the zero-shot forecaster's past errors*

- **Scale 1.29.** Errors spread 29% wider than the bands say, as Day 2's 74% coverage warned.
- **Across products 0.09.** Same-day errors of two products move together a little.
- **Across days 0.04.** Within a week, one product's days are nearly independent.

#### [day3#stress-result] Independence with face-value bands is the optimistic corner

*Lab 3.4, the course run, 2,000 weeks each, the plan of Lab 3.2*

| Scenarios | Mean shortage probability | 3 or more in-stock products short | Fill rate | Shortage cost, bad week (95th pct) |
| --- | --- | --- | --- | --- |
| Independent, bands as given | 32% | 27% | 90.8% | SAR 11,544 |
| As measured | 38% | 49% | 88.5% | SAR 23,829 |
| Stress (shared shock 0.3) | 41% | 46% | 86.1% | SAR 39,343 |

P003, cut to fit one truck, runs short in every scenario (fill rate 12%).

#### [day3#sample-average] Optimise on the scenarios themselves: sample average approximation

*When the risk is too high*

- **Today's plan.** One target per product from the forecast, then a programme that charges distance from the target.
- **Sample average.** Shortfall and leftover variables for each of 200 scenarios; minimise the average cost under the same capacity.

#### [day3#checkpoint-4] How likely are two shortages together?

*Checkpoint*

**Question:** Rice and sugar each run short in 10% of weeks. A planner reports a 1% chance that both run short in the same week. When is that report most wrong?

- A. When a shared shock, such as a pre-Eid rush, lifts both demands in the same week
- B. When both come from the same supplier
- C. When each product's days are independent of each other
- D. When the forecaster's bands are too wide

#### [day3#lab-3-4] Stress-test the plan

Lab 3.4 · 25 min

Stress-test the plan with simulated demand and report shortage risk: 2,000 scenarios, correlated and independent, shortage probability and fill rate per product.

[D3_4_stress_test_plan.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D3_4_stress_test_plan.ipynb)

- [ ] Scenarios from the forecast (6 min)
- [ ] Run the plan through them (6 min)
- [ ] Shortage risk per product (6 min)
- [ ] Correlated against independent (7 min)
- [ ] ✱ Stretch: Re-solve the plan on 200 scenarios and compare the risk

#### [day3#wrap-up] Judge forecasts by the decisions they cause, and report the plan's risk

*Day 3 · Three things to keep*

- **Order the quantile.** The critical ratio, not the mean; dairy below the median.
- **Decide together.** Trucks and space make orders compete; price a constraint by re-solving.
- **Regret and risk.** Rank forecasters in riyals, with an interval; stress the plan before signing.

#### [day3#next] Shift and adoption

*Tomorrow · Day 4*

The northern region changes the data under the model: detecting the shift, validating as you will deploy, correcting intervals, and the licence, cost and residency conditions of adoption.

- Unfinished labs from today can be finished at home; every Day 4 lab starts from supplied files.
- Skim the Personal Data Protection Law's transfer regulation summary in the course README.

## Day 4: Shift and adoption

Shift found, validated and corrected, then the adoption decision

#### [day4#title] Shift and adoption

*SDAIA Academy · Advanced Modelling for Structured Data*

Day 4: Shift found, validated and corrected, then the adoption decision

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-404

#### [day4#agenda] Shift and adoption

*Today*

- **Session 1 · Detecting shift**: Train a classifier to tell old depots from the new region: if it can, the data has shifted
- **Session 2 · Validating under shift**: The validation scheme that matches the deployment shift is the one whose error you can promise
- **Session 3 · Correcting for shift**: Weighting calibration rows by how deployment-like they are restores interval coverage under covariate shift
- **Session 4 · Adoption conditions**: A model is adopted when accuracy, cost, licence and the data's classification and residency all allow it

#### [day4#review-day3] What do you put in front of Ahlam?

*Review of Day 3*

**Question:** Over twelve weeks LightGBM's orders cost SAR 2,972 a week less than the zero-shot forecaster's, with a 95% interval from -7,820 to +506. Its scaled quantile loss is clearly lower. What do you report?

- A. LightGBM is better: its error is lower
- B. LightGBM probably costs less, but twelve weeks cannot separate the two; keep both on test
- C. The two forecasters cost the same
- D. Smoothing, because it won at one origin

### Session 1: Detecting shift

#### [day4#s1] Detecting shift

*Session 1*

- Name covariate, label and concept shift and a test for each
- Run adversarial validation between training and deployment data
- Read which features carry the shift

#### [day4#three-shifts] Inputs, outcomes or the link between them can move, and each needs its own test

*What can change between training and deployment*

| Shift | What changes | At Bustan's north | Test |
| --- | --- | --- | --- |
| **Covariate** | the features, p(x) | longer distances, new drivers | adversarial validation |
| **Label** | the outcome rate, p(y) | more late deliveries | compare rates |
| **Concept** | the link, p(y \| x) | distance costs more on new routes | old model on new outcomes |

#### [day4#adversarial-validation] Train a classifier to tell the old depots from the north: if it can, the inputs have shifted

*A test that needs no labels*

1. **Stack** old orders and northern orders
2. **Label** north = 1, old = 0
3. **Classify** grouped by customer, out of fold
4. **Read** AUC and the features it used

> **Same period on both sides.** Old orders from before the north opened would let the classifier separate by date.

#### [day4#what-moved] The classifier separates the north almost perfectly, on driver tenure and distance

*Lab 4.1, the course run, orders since 24 August 2025*

| Feature | Share of the classifier's gain | Old depots, median | North, median |
| --- | --- | --- | --- |
| Driver tenure | 57% | 28 months | 8 months |
| Distance | 32% | 17.2 km | 44.1 km |
| Truck utilisation | 5% | 0.68 | 0.56 |

Adversarial AUC 0.954, folds grouped by customer. Grocery is 66% of northern orders, 47% of old.

#### [day4#label-concept] The new inputs explain less than half of the extra lateness

*Lab 4.1, separating the shifts*

| Item | Value |
| --- | --- |
| Old depots, observed | 18.6% |
| North, predicted by the old model | 29.2% |
| North, observed | 43.6% |

Covariate shift accounts for +10.6 points; the other +14.4 points are a changed link between the features and lateness.

#### [day4#checkpoint-1] What does an adversarial AUC of 0.95 tell Shadi?

*Checkpoint*

**Question:** The adversarial classifier separates northern orders from old ones with AUC 0.95, mostly on driver tenure and distance. What can you conclude before any northern outcome is known?

- A. The late model will fail in the north
- B. Drop tenure and distance from the late model
- C. Many northern orders have few look-alikes in the training data, so its predictions there are untested
- D. The north has a higher late rate

#### [day4#lab-4-1] Adversarial validation

Lab 4.1 · 25 min

Run adversarial validation between old depots and the new region: a classifier that separates them, its grouped AUC, the features that carry the shift, and the late rate and its link to distance on each side.

[D4_1_run_adversarial_validation.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D4_1_run_adversarial_validation.ipynb)

- [ ] Two samples: old depots and the north (3 min)
- [ ] The adversarial classifier (8 min)
- [ ] Which features carry the shift (6 min)
- [ ] Label and concept shift, checked apart (8 min)
- [ ] ✱ Stretch: Drop the top shifted feature and run again

### Session 2: Validating under shift

#### [day4#s2] Validating under shift

*Session 2*

- Run random, grouped, temporal and region-held-out validation on one model
- Compare each estimate with the error in the new region
- Report error by region and customer segment before and after shift

#### [day4#four-schemes] Each validation scheme answers a different question about the future

*Validating on the old depots, before the north has outcomes*

| Scheme | Test rows | Answers |
| --- | --- | --- |
| Random | shuffled orders | new orders from the same customers |
| Grouped | unseen customers | new customers in the same regions |
| Temporal | the last 20% of days | the next months, same customers |
| Region-held-out | one unseen old region | a region the model has not seen |

#### [day4#honest-scheme] Region-held-out came within 0.01 AUC of the north; no scheme saw the calibration fail

*Lab 4.2, the course run, one LightGBM, 72,000 old orders*

| Scheme | AUC | Log loss | Predicted late rate | Observed |
| --- | --- | --- | --- | --- |
| Random | 0.871 | 0.316 | 18.0% | 18.1% |
| Temporal | 0.881 | 0.316 | 19.5% | 19.2% |
| Grouped | 0.773 | 0.384 | 18.1% | 18.0% |
| Region-held-out | 0.770 | 0.389 | 17.9% | 18.3% |
| **Deployed in the north** | **0.763** | **0.618** | **31.0%** | **43.6%** |

#### [day4#subgroup-error] The average hides a segment the model barely ranks

*Lab 4.2, deployed in the north, by segment*

| Segment | Orders | AUC | Predicted | Observed |
| --- | --- | --- | --- | --- |
| Grocery | 5,242 | 0.764 | 29.7% | 42.0% |
| HoReCa | 1,194 | 0.817 | 39.2% | 56.5% |
| Hypermarket | 608 | 0.837 | 34.3% | 29.3% |
| Institutional | 956 | 0.651 | 25.3% | 45.6% |

#### [day4#checkpoint-2] Which promise can you make Shadi?

*Checkpoint*

**Question:** Before the north opened, the analyst reported "AUC 0.87, well calibrated" from random 5-fold validation on the old depots. Knowing what you know now, what should the report have said?

- A. A region-held-out AUC near 0.77, and no promise on calibrated probabilities until northern outcomes arrive
- B. AUC 0.87, since random folds use the most data
- C. A temporal AUC of 0.88, since the north is later in time
- D. No estimate at all: nothing measured on old data applies to the north

#### [day4#lab-4-2] Four schemes against deployed error

Lab 4.2 · 25 min

Compare random, grouped and temporal validation with deployed error: the same model under four schemes, the error later measured in the north, and error by region and segment.

[D4_2_compare_random_grouped.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D4_2_compare_random_grouped.ipynb)

- [ ] One model, four schemes (8 min)
- [ ] Deployed error in the north (5 min)
- [ ] Which scheme was honest (5 min)
- [ ] Error by region and segment (7 min)
- [ ] ✱ Stretch: Validate on Hail only and predict Tabuk

### Session 3: Correcting for shift

#### [day4#s3] Correcting for shift

*Session 3*

- Recall split conformal intervals and the exchangeability they assume
- Estimate density-ratio weights from an adversarial classifier
- Apply weighted conformal intervals and check coverage in the new region

#### [day4#recap-conformal] Split conformal holds 90% of the time only if new orders are exchangeable with the calibration orders

*Recap from DS-305, then what breaks*

```python
# file: labs/_src/D4_3_apply_weighted_conformal.py
scores = np.abs(y_cal - model.predict(X_cal))   # log hours
k = int(np.ceil((n + 1) * 0.9))
q = np.sort(scores)[k - 1]                      # interval: prediction ± q
```

> Lab 4.1's adversarial AUC of 0.95 says the northern orders are not exchangeable with the old ones.

#### [day4#density-ratio-weights] Weight each calibration order by how northern it looks

*Correcting for covariate shift*

1. **Classifier** p(north | x), out of fold
2. **Ratio** w(x) = p / (1 - p)
3. **Weighted quantile** of calibration scores, plus the new order's own weight at infinity

> **Takeaway.** If too few calibration orders look like a new one, its interval is infinite: an honest "we do not know".

#### [day4#coverage-result] Weighting restores 90% overall, and leaves institutional customers short

*Lab 4.3, the course run, 90% intervals for delivery hours in the north*

| Northern orders | Split conformal | Weighted |
| --- | --- | --- |
| All | 87.7% | 90.1% |
| Grocery | 89.0% | 91.4% |
| HoReCa | 85.5% | 88.2% |
| Hypermarket | 95.2% | 96.5% |
| Institutional | 78.7% | 81.0% |

Median width 61 hours split, 67 weighted. Effective sample size 1,082 of 22,325 calibration orders.

#### [day4#checkpoint-3] What did the weights correct?

*Checkpoint*

**Question:** After weighting, coverage in the north is 90.1% overall but 81.0% for institutional customers. Which statement is right?

- A. The weights are wrong: weighted conformal guarantees 90% in every segment
- B. Institutional customers are too few to measure coverage
- C. Raising the institutional weights would fix it
- D. Weights correct a change in the inputs; the institutional gap is a change in the link, which needs northern outcomes

#### [day4#lab-4-3] Weighted conformal intervals

Lab 4.3 · 25 min

Apply weighted conformal intervals and check coverage in the new region: delivery-hours intervals calibrated on old depots, weights from the adversarial classifier, coverage by region and segment.

[D4_3_apply_weighted_conformal.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D4_3_apply_weighted_conformal.ipynb)

- [ ] Split conformal on old depots (6 min)
- [ ] Coverage in the north (4 min)
- [ ] Weights from the classifier (7 min)
- [ ] Weighted intervals and their coverage (8 min)
- [ ] ✱ Stretch: Clip the weights and watch width and coverage

### Session 4: Adoption conditions

#### [day4#s4] Adoption conditions

*Session 4*

- Check the licence of a model's weights and outputs against the intended use
- Apply the classification level and the transfer rules to a deployment option
- Write an adoption decision with its conditions and choose the capstone task

#### [day4#four-checks] A model is adopted when accuracy, licence, data and cost all allow it

*What adoption requires*

- **Accuracy.** Against a tuned baseline, with intervals, under the deployment's shift, by subgroup.
- **Licence.** Weights and outputs allowed for commercial production use.
- **Data.** Classification, personal data, where it may be processed.
- **Cost.** Running and monitoring every cycle, against the gain in riyals.

#### [day4#licences] Three candidates allow commercial use; TabPFN-2.6 restricts even its outputs

*Checked on the official model pages, November 2026*

| Model | Task | Weights licence | Production at Bustan |
| --- | --- | --- | --- |
| TabICLv2 | tables | BSD-3-Clause | allowed |
| TabPFN-2.6 | tables | TabPFN-2.6 Non-Commercial License | blocked: outputs for non-commercial use only |
| Chronos-2 | series | Apache-2.0 | allowed |
| TimesFM 2.5 | series | Apache-2.0 | allowed |

#### [day4#residency] Orders hold personal data: leaving the Kingdom needs a purpose, a risk assessment and safeguards

*The data decides where the model may run*

| Dataset (Bustan's register) | Level | Personal data | Outside the Kingdom |
| --- | --- | --- | --- |
| Orders with customer ids | Restricted | yes: sole traders | PDPL Article 29 conditions, transfer risk assessment, safeguards |
| Daily demand by product and depot | Restricted | no | Bustan's policy: an approved exception |

Self-hosting in Riyadh is allowed for every candidate. Google Colab counts as outside the Kingdom: fine for synthetic course data only.

#### [day4#adoption-table] One go with conditions, two not now, one no-go

*Lab 4.4, the course's adoption table*

| Model | Decision | Reason and condition |
| --- | --- | --- |
| TabICLv2 | not now | the course run cannot tell it from tuned LightGBM; retest with the real model |
| TabPFN-2.6 | no-go | non-commercial licence |
| Chronos-2 | go with conditions | self-hosted in Riyadh, for holiday windows, if its stacking weight pays its cost |
| TimesFM 2.5 | not now | not evaluated in this course |

Running costs from the synthetic sheet: SAR 2 to 84 a month per option, small next to SAR 3,000 a week of regret between two forecasters.

#### [day4#checkpoint-4] Which option can Bustan use for late-delivery risk?

*Checkpoint*

**Question:** A cloud reseller offers the open TabPFN-2.6 weights as a hosted service outside the Kingdom, at SAR 6 per 1,000 predictions, with a strong benchmark score. The task reads Bustan's orders with customer ids. What blocks it first?

- A. The price per prediction
- B. The licence: its outputs are for non-commercial use only
- C. The benchmark score is not high enough
- D. Only the transfer of personal data outside the Kingdom

#### [day4#lab-4-4] Licence, cost and residency

Lab 4.4 · 25 min

Complete a licence, cost and residency table and choose the capstone task: two candidates per task, Bustan's classification register, deployment options and a go or no-go with reasons.

[D4_4_complete_licence_cost.ipynb](https://sdaiaacademy.github.io/ds-404/labs/D4_4_complete_licence_cost.ipynb)

- [ ] The candidates and their licences (6 min)
- [ ] Classification and residency (7 min)
- [ ] Cost per run (5 min)
- [ ] Adoption table and memo lines (4 min)
- [ ] Choose the capstone task (3 min)
- [ ] ✱ Stretch: Write the evaluation protocol as a checklist another analyst could rerun

#### [day4#wrap-up] Validate as you will deploy, and adopt only under conditions you can name

*Day 4 · Three things to keep*

- **Detect.** Adversarial validation on the inputs; rates and an old model on the outcomes.
- **Validate and correct.** Hold out what deployment holds out; weight for covariate shift, recalibrate for the rest.
- **Adopt with conditions.** Licence, data, cost and subgroup error, in one table.

#### [day4#next] Capstone

*Tomorrow · Day 5*

An evaluation protocol and an adoption decision for one structured-data task, defended in 10 minutes with 10 minutes of questions.

- Start from labs/capstone/: the starter and the template.
- Bring the numbers with their intervals, and the run that produced each.

## Capstone: Capstone project presentations

An evaluation protocol and an adoption decision, defended before two assessors

#### [capstone#title] Capstone project presentations

*SDAIA Academy · Advanced Modelling for Structured Data*

Capstone: An evaluation protocol and an adoption decision, defended before two assessors

Data Science Track · 4 teaching days + capstone day · 10 min talk and demo · 10 min Q&A · DS-404

#### [capstone#agenda] How the presentations run

*Capstone*

- **Expectations and final prep**: the rubric, a last rehearsal, then the first presentations
- **Presentations**: each one timed, with questions after it
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#rubric] The rubric

*How presentations are scored*

| Item | Value |
| --- | --- |
| Fair comparison | 25% |
| Validation under shift | 25% |
| Ensemble or decision analysis | 20% |
| Adoption reasoning | 15% |
| Reproducibility | 15% |

> **Show results from at least two measured runs.** A claim without a number is a plan, not a result.

#### [capstone#format] Ten minutes to defend the decision, ten minutes of questions from two assessors

*Format*

- **1 · The task and the decision.** Which task, who decides, what is decided. One slide.
- **2 · The protocol.** The tuned baseline and its budget, two pretrained candidates, the ensemble, the validation scheme and the shift it matches.
- **3 · Results and conditions.** Every number with its interval and its run; regret where a decision follows; the adoption decision with its conditions.

> **Peer feedback.** After each defence, everyone writes one strength, one question and one suggestion.

#### [capstone#shift-question] An assessor names a new shift; you predict which result changes before you re-run

*Inside the ten minutes of questions*

1. **Scenario** e.g. a third region, a supplier change, Ramadan inside the test window
2. **Prediction** which validation number moves, and which way
3. **Re-run** the protocol on the new split, live or right after

> **What earns the marks.** A prediction with a reason, then an honest comparison with the re-run, whichever way it went.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Project |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first presentation*

Rehearse once, record a backup, try the demo on this screen.

#### [capstone#now-presenting] Now presenting

*Now presenting*

#### [capstone#review] What the presentations showed

*Trainer review*

- **Strongest patterns.** What several projects got right.
- **Common gaps.** What most projects would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **Your project stays yours:** keep it reproducible, and re-run the protocol when the data or the candidates change
- **The follow-up task, within three weeks:** apply the evaluation protocol to one workplace or public dataset and report the measured results to the trainer
- **Next in the Academy:** no catalog course follows DS-404; DS-304 and DS-305 are its foundations if a colleague needs them

#### [capstone#thanks] Thank you

*SDAIA Academy*

Advanced Modelling for Structured Data
