---
course: "Advanced Modelling for Structured Data"
code: "DS-404"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-404/slides/day4.html"
---

# Advanced Modelling for Structured Data

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
