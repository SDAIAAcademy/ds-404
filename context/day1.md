---
course: "Advanced Modelling for Structured Data"
code: "DS-404"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-404/slides/day1.html"
---

# Advanced Modelling for Structured Data

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
