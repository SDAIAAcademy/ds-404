# Advanced Modelling for Structured Data: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Tuned boosted trees first | [`D1_1_tune_boosted_tree.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D1_1_tune_boosted_tree.ipynb) |
| Day 1 | Session 2 · In-context learning on tables | [`D1_2_run_pretrained_tabular.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D1_2_run_pretrained_tabular.ipynb) |
| Day 1 | Session 3 · Learning curves decide | [`D1_3_plot_both_across.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D1_3_plot_both_across.ipynb) |
| Day 1 | Session 4 · Reading a leaderboard claim | [`D1_4_check_claim_unequal.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D1_4_check_claim_unequal.ipynb) |
| Day 2 | Session 1 · Zero-shot against a benchmark floor | [`D2_1_score_zero_shot.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D2_1_score_zero_shot.ipynb) |
| Day 2 | Session 2 · Where zero-shot loses | [`D2_2_find_series_groups.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D2_2_find_series_groups.ipynb) |
| Day 2 | Session 3 · Holidays that move | [`D2_3_add_ramadan_eid.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D2_3_add_ramadan_eid.ipynb) |
| Day 2 | Session 4 · Ensembles worth their cost | [`D2_4_stack_forecasters_rolling.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D2_4_stack_forecasters_rolling.ipynb) |
| Day 3 | Session 1 · Asymmetric costs | [`D3_1_compute_newsvendor_quantities.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D3_1_compute_newsvendor_quantities.ipynb) |
| Day 3 | Session 2 · Optimising under capacity | [`D3_2_solve_order_truck.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D3_2_solve_order_truck.ipynb) |
| Day 3 | Session 3 · Scoring by regret | [`D3_3_rank_three_forecasters.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D3_3_rank_three_forecasters.ipynb) |
| Day 3 | Session 4 · Uncertainty in the plan | [`D3_4_stress_test_plan.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D3_4_stress_test_plan.ipynb) |
| Day 4 | Session 1 · Detecting shift | [`D4_1_run_adversarial_validation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D4_1_run_adversarial_validation.ipynb) |
| Day 4 | Session 2 · Validating under shift | [`D4_2_compare_random_grouped.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D4_2_compare_random_grouped.ipynb) |
| Day 4 | Session 3 · Correcting for shift | [`D4_3_apply_weighted_conformal.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D4_3_apply_weighted_conformal.ipynb) |
| Day 4 | Session 4 · Adoption conditions | [`D4_4_complete_licence_cost.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-404/blob/main/labs/D4_4_complete_licence_cost.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
