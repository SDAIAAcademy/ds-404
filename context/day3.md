---
course: "Advanced Modelling for Structured Data"
code: "DS-404"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-404/slides/day3.html"
---

# Advanced Modelling for Structured Data

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
