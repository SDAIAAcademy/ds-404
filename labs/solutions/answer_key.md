# Answer key

For trainers. The answers to the checkpoint questions at the end of every lab notebook, and the letters of the right options in the decks' checkpoints. The reference numbers each lab produced are in `labs/data/reference/<lab>.json`.

## Deck checkpoints

| Deck | Slide | Right option |
|---|---|---|
| day1 | checkpoint-1 | B: 0.78, because the folds keep each customer on one side |
| day1 | checkpoint-2 | C: The 103 fits the baseline's tuning cost before that fit |
| day1 | checkpoint-3 | A: No difference detectable at 500 orders with three repeats |
| day1 | checkpoint-4 | D: The boosted trees were run at default settings |
| day2 | review-day1 | C: More repeats at 500 and 1,000 orders whose paired interval stays above zero |
| day2 | checkpoint-1 | A: Not as a replacement; test where it wins and what it adds to an ensemble first |
| day2 | checkpoint-2 | D: Use the tuned LightGBM, which wins the group, and retest the pretrained model later |
| day2 | checkpoint-3 | B: 357 days, 51 whole weeks |
| day2 | checkpoint-4 | C: The weights saw the windows they are scored on |
| day3 | review-day2 | C: What 0.4% less loss is worth in riyals, against the cost of running four models |
| day3 | checkpoint-1 | D: The 0.25 quantile |
| day3 | checkpoint-2 | B: Decline: re-solving with three reefers would not change the plan, because dairy already fits |
| day3 | checkpoint-3 | C: Its errors fell where they cost most: dairy overage and the Eid al-Adha week |
| day3 | checkpoint-4 | A: When a shared shock, such as a pre-Eid rush, lifts both demands in the same week |
| day4 | review-day3 | B: LightGBM probably costs less, but twelve weeks cannot separate the two; keep both on test |
| day4 | checkpoint-1 | C: Many northern orders have few look-alikes in the training data, so its predictions there are untested |
| day4 | checkpoint-2 | A: A region-held-out AUC near 0.77, and no promise on calibrated probabilities until northern outcomes arrive |
| day4 | checkpoint-3 | D: Weights correct a change in the inputs; the institutional gap is a change in the link, which needs northern outcomes |
| day4 | checkpoint-4 | B: The licence: its outputs are for non-commercial use only |

## Lab 1.1 · A tuned baseline under a budget

**Questions**

1. A colleague reports 0.84 AUC from random 5-fold CV on the same orders. What do you tell him?
2. Your search took 100 fits. A pretrained model will be run once. Which numbers must the comparison table
   carry so that it is fair?
3. Why are the three configurations averaged on probabilities, not on predicted classes?

**Answers**

1. Random folds put a customer's orders on both sides, and the model recognises customers by fixed fields
   (distance, segment). New customers are what Bustan predicts for: use grouped folds or held-out customers.
2. The tuning budget in fits and seconds, the fit and predict time of each model, and the same folds and
   metric for both.
3. AUC and the cost decisions of Day 3 use the ranking and the probability; averaging classes throws both away.

## Lab 1.2 · A pretrained model on the same folds

**Questions**

1. The pretrained model's `fit` takes a fraction of a second and its `predict` takes longer. Why?
2. Your table shows the two AUCs within 0.01 and the interval of the difference crosses zero. What may you
   claim?
3. Why must both models use the same folds rather than each its own random split?

**Answers**

1. An in-context learner does its work at prediction time: it attends from each test row to the training rows.
2. Only that this experiment cannot tell them apart at 2,000 orders; Lab 1.3 asks the same question at
   other sizes.
3. Paired folds remove the split-to-split noise from the difference; separate splits add it back.

## Lab 1.3 · Learning curves with intervals

**Questions**

1. At 500 orders the pretrained model's mean is higher but the interval of the difference crosses zero.
   What do you write in the memo?
2. Why are the repeats drawn as new subsamples instead of new folds of one subsample?
3. Why are the 50,000-order scores supplied rather than computed in the lab?

**Answers**

1. That the two cannot be told apart at that size with this evidence, and how many repeats would decide it.
2. The question is how the method does on a sample of that size, so the sample itself must vary.
3. An in-context model at 50,000 rows needs a GPU and more than the lab's minutes; the file says which model
   wrote it.

## Lab 1.4 · Check a leaderboard claim

**Questions**

1. Which single column of the table voids the "35 of 36" most?
2. A dataset is public and the vendor's model card lists it among its validation suites. Keep it?
3. The clean small-data slice favours the pretrained model. What would make you act on it at Bustan?

**Answers**

1. `gbm_trials_as_claimed`: zero tuning trials against a model that ensembles eight passes.
2. No: it was used to select the model, so it cannot also judge it.
3. The same sign on Bustan's own curve at Bustan's sizes, with an interval that excludes zero.

## Lab 2.1 · Zero-shot against the supplied backtests

**Questions**

1. The zero-shot forecaster beats the floor but not the tuned LightGBM. Is it useless?
2. Why is every series divided by its own scale before averaging?
3. A method has the lowest quantile loss and 65% coverage of its 80% band. What do you report?

**Answers**

1. No: it needed no training, no features and no tuning. Its value depends on where it wins (Lab 2.2) and on
   what it adds to an ensemble (Lab 2.4).
2. Otherwise the fast-moving staples, in thousands of cases, decide the average alone.
3. Both numbers: a low loss with bands that are too narrow; Day 3 shows what narrow bands cost.

## Lab 2.2 · Where the benchmark wins

**Questions**

1. Over all series the zero-shot forecaster is second best. In the short group it ties smoothing and trails
   the tuned LightGBM most. What do you tell the planner who manages new products?
2. Why is the routing rule chosen on earlier origins?
3. A group holds 12 series. Its difference is large but the interval crosses zero. Act on it?

**Answers**

1. For new products the tuned LightGBM is the method to beat; the zero-shot model is no better than smoothing
   there. Test the real pretrained model in Colab before changing anything.
2. A rule picked on the scored windows is tested on its own training data.
3. Not yet: gather more origins first, or treat it as a hypothesis for the capstone.

## Lab 2.3 · Ramadan and Eid as covariates

**Questions**

1. Why does a 364-day lag miss Ramadan, and why does a 354-day lag not fix it alone?
2. Is it leakage to give the forecaster the Eid dates of the next 28 days?
3. Covariates help the zero-shot model on holiday days. What do you check on ordinary days?

**Answers**

1. Ramadan follows the Hijri year of about 354 days, so it moves about eleven days earlier each Gregorian year;
   354 days later is a different weekday, so the weekly pattern breaks. 357 days (51 weeks) keeps both close.
2. No: the calendar is published in advance. Leakage is anything not known at the origin, such as next week's sales.
3. That the ordinary-day error did not get worse: a covariate that adds noise elsewhere is not free.

## Lab 2.4 · Stacking on rolling-origin windows

**Questions**

1. Why are the weights for origin 2026-03-15 fitted on the four earlier origins only?
2. The stack beats the best single model by 2% with an interval from -0.5% to 4.5%. What do you recommend?
3. Why choose the "best single model" on earlier windows too?

**Answers**

1. Their windows end on or before 2026-03-15 (2026-02-15's reaches it exactly); a later window would hold the
   days being scored.
2. Keep the single model, or rerun with more origins: the gain is not distinguishable from zero and the stack
   costs four models.
3. Picking it on the test windows gives the single model an advantage the stack does not get.

## Lab 3.1 · Newsvendor quantities

**Questions**

1. A product's critical ratio is 0.27. Order above or below the median?
2. Why is the weekly spread not seven times the daily spread?
3. The forecaster's 80% bands held only 74% of the time on Day 2. What does that do to these orders?

**Answers**

1. Below: a leftover costs more than a shortage, so the 0.27 quantile.
2. Independent days add variances, not spreads: the weekly sd is the square root of the sum of squares.
3. Bands too narrow make high quantiles too low: dry goods run short more often than planned.

## Lab 3.2 · Orders and trucks for one depot week

**Questions**

1. Why are trucks integer variables, not continuous ones?
2. Depot space binds. Imad offers to rent 10 cubic metres for SAR 500 for the week. Accept?
3. Why does the programme cut dry goods rather than dairy when space is short?

**Answers**

1. Half a truck costs a whole trip; a continuous truck count would underprice small orders.
2. Compare SAR 500 with the saving the re-solve printed.
3. Cutting dry goods does two things at once: it frees depot space and lets supplier S1 fit one truck instead
   of two (SAR 2,400). Dairy already fits its two reefers, so cutting it saves space only.

## Lab 3.3 · Regret against error

**Questions**

1. Why is regret equal to realised cost in this lab, and when would it not be?
2. A forecaster has lower error but higher regret at one origin. Name one reason.
3. Ahlam asks which forecaster to adopt. What do you report besides the overall ranking?

**Answers**

1. With no capacity limit, the hindsight order equals the demand and costs nothing. Under the depot limit,
   the best plan in hindsight still leaves some demand unmet, so regret is smaller than cost.
2. Its errors fall where they cost most: too high on chilled dairy, where a leftover loses 60% of its cost,
   or too low on dry goods ordered at the 0.9 quantile.
3. The paired difference per week with its interval, and the regret by category: a decision should not
   rest on a gap the weeks cannot separate from noise.

## Lab 3.4 · Stress-test the plan

**Questions**

1. Two products each run short 10% of weeks. When is the chance both run short the same week far above 1%?
2. Why does a scale above 1 matter more for dry goods ordered at the 0.9 quantile than for dairy?
3. What do you add to the plan before Ahlam signs it?

**Answers**

1. When their demand shares a shock (the same week, the same weather, the same holiday): independence
   multiplies the probabilities; correlation does not.
2. A high quantile sits far in the tail, where a narrow band moves it most; a low quantile sits near the middle.
3. The shortage probability and fill rate per product under the measured error scale, and the chance that
   several products run short together.

## Lab 4.1 · Adversarial validation

**Questions**

1. The adversarial AUC is 0.95. Does that mean the late model will fail in the north?
2. Why are the folds grouped by customer?
3. Which of the three shifts can importance weighting correct?

**Answers**

1. Not by itself: it says the inputs differ. Whether the model fails is measured on northern outcomes (Lab 4.2).
2. Customers belong to one region; random folds would let the classifier recognise customers it has seen.
3. Covariate shift, where the link from features to outcome holds. Not concept shift.

## Lab 4.2 · Four schemes against deployed error

**Questions**

1. Which two schemes leaked, and how?
2. Region-held-out came closest on AUC. Why did it still miss the log loss?
3. What do you promise Shadi before the north has outcomes?

**Answers**

1. Random and temporal: the same customers' orders were in training and test.
2. The old regions resemble each other; the north differs in its late rate and its distance effect, which no
   old region shows.
3. The region-held-out AUC as an upper estimate of ranking, no calibrated probabilities until northern outcomes
   are in, and a date to recalibrate.

## Lab 4.3 · Weighted conformal intervals

**Questions**

1. What assumption does split conformal need, and which Lab 4.1 result breaks it?
2. Why are the classifier's probabilities scored out of fold?
3. Coverage is back near 90% overall. What do you still tell Shadi?

**Answers**

1. Calibration and new orders exchangeable; the adversarial AUC of 0.95 says they are not.
2. In-sample probabilities are overconfident near 0 and 1; their ratios explode into a few huge weights.
3. That the guarantee is overall, not per segment: institutional orders still fall outside more often, until
   northern outcomes recalibrate the intervals.

## Lab 4.4 · Licence, cost and residency

**Questions**

1. A candidate's weights are Apache-2.0. Is the adoption question settled?
2. Why is Google Colab treated as outside the Kingdom?
3. Which of the four checks blocked a candidate outright in this lab?

**Answers**

1. No: the data, its residency and the evidence of accuracy still decide.
2. The user does not choose where the machine runs, so Bustan cannot show the data stayed in the Kingdom.
3. The licence: TabPFN-2.6's outputs are for non-commercial use only.

