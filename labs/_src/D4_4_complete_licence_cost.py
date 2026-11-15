# %% [markdown]
# # Lab 4.4 · Licence, cost and residency
#
# Day 4 · Session 4 (Adoption conditions) · 25 min.
#
# Ahlam will adopt a pretrained model only with conditions she can defend. For two tasks (late-delivery risk on
# tables, daily demand on series) there are two candidates each. You check each candidate's licence against
# production use, the data's classification and the transfer rules against each deployment option, the cost of a
# month of runs, and you write the adoption table and its memo lines. Then you choose your capstone task.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/adoption/candidates.json` (licences checked on the official model pages in
# November 2026), `data/adoption/classification_register.csv` (Bustan's register, kept by Asil Al-Rajeh),
# `data/adoption/cost_sheet.csv` (synthetic rates) and the results of Days 1 and 2 (`data/reference/`).
#
# **Tasks**
# - [ ] 1 · The candidates and their licences (6')
# - [ ] 2 · Classification and residency (7')
# - [ ] 3 · Cost per run (5')
# - [ ] 4 · Adoption table and memo lines (4')
# - [ ] 5 · Choose the capstone task (3')
# - [ ] ✱ Stretch · Write the evaluation protocol as a checklist another analyst could rerun

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-404.git"
    if not Path("/content/ds-404").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-404"], check=True)
    os.chdir("/content/ds-404/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import pandas as pd
import bustan as U

pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 70)
cands = pd.DataFrame(U.js("adoption/candidates.json"))
register = pd.read_csv(U.DATA / "adoption" / "classification_register.csv")
rates = pd.read_csv(U.DATA / "adoption" / "cost_sheet.csv")
print(cands[["task", "model", "weights_licence", "commercial_use", "outputs_restricted"]].to_string(index=False))

# %% [markdown]
# ## 1 · The candidates and their licences (6')
#
# Bustan's use is commercial: the predictions steer orders and deliveries that earn money. A licence allows
# production only if the weights may be used commercially **and** the outputs are not restricted.
#
# ✏️ **Exercise** · Add a column `licence_ok` (True or False) and a column `licence_note` that says why.

# %%
def licence_check(c: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    ok = c.commercial_use.str.startswith("yes") & c.outputs_restricted.str.startswith("no")
    note = c.weights_licence.where(ok, c.weights_licence + ": " + c.commercial_use + "; " + c.outputs_restricted)
    return c.assign(licence_ok=ok, licence_note=note)
    # END SOLUTION

# %%
try:
    cands = licence_check(cands)
except NotImplementedError:
    ok = cands.commercial_use.str.startswith("yes") & cands.outputs_restricted.str.startswith("no")
    cands = cands.assign(licence_ok=ok, licence_note=cands.weights_licence.where(ok, cands.weights_licence + ": " + cands.commercial_use + "; " + cands.outputs_restricted))
print(cands[["model", "licence_ok", "licence_note"]].to_string(index=False))
U.check(cands.set_index("model").licence_ok.to_dict() == {"TabICLv2": True, "TabPFN-2.6": False, "Chronos-2": True, "TimesFM 2.5": True},
        "one candidate is blocked by its licence")

# %% [markdown]
# ## 2 · Classification and residency (7')
#
# Each task reads one dataset of the register:
#
# - late-delivery risk reads **orders (with customer ids, depot, distance)**: Restricted, personal data;
# - daily demand reads **daily demand by product and depot**: Restricted, no personal data.
#
# The rules Bustan applies:
#
# - **Personal data leaving the Kingdom.** The Personal Data Protection Law (Article 29) allows a transfer only for
#   listed purposes, with an adequate level of protection and the minimum data needed; the Regulation on Personal
#   Data Transfer outside the Kingdom requires a risk assessment before it (Article 7) and appropriate safeguards
#   (standard contractual clauses, binding common rules or a certificate) in the exempted cases (Article 4).
# - **Restricted data.** Bustan's policy, which follows the levels of the National Data Governance Policies, keeps
#   Restricted data on systems Bustan controls in the Kingdom unless Asil approves an exception.
# - Any option whose location the user does not choose (such as Google Colab) counts as outside the Kingdom; it is
#   fine for the course's synthetic data and nothing else.
#
# ✏️ **Exercise** · For every candidate and deployment option, add `residency` with one of `"allowed"`,
# `"conditions: ..."` or `"not allowed: ..."`.

# %%
DATASET = {"late-delivery risk (tables)": "orders (with customer ids, depot, distance)",
           "daily demand (series)": "daily demand by product and depot"}
rows = cands.explode("deployment").reset_index(drop=True)
rows["dataset"] = rows.task.map(DATASET)
rows = rows.merge(register, left_on="dataset", right_on="dataset", how="left")


def residency(r) -> str:
    # BEGIN SOLUTION
    inside = "in Riyadh" in r.deployment
    if inside:
        return "allowed"
    if r.personal_data == "yes":
        return ("not allowed: personal data outside the Kingdom needs a PDPL Article 29 purpose, a transfer risk "
                "assessment and safeguards; Restricted data also needs Asil's exception")
    if r.level == "Restricted":
        return "conditions: Restricted data outside Bustan's systems needs Asil's written exception"
    return "allowed"
    # END SOLUTION

# %%
try:
    rows["residency"] = rows.apply(residency, axis=1)
except NotImplementedError:
    def residency(r):
        if "in Riyadh" in r.deployment:
            return "allowed"
        if r.personal_data == "yes":
            return "not allowed: personal data outside the Kingdom needs a PDPL Article 29 purpose, a transfer risk assessment and safeguards; Restricted data also needs Asil's exception"
        return "conditions: Restricted data outside Bustan's systems needs Asil's written exception" if r.level == "Restricted" else "allowed"
    rows["residency"] = rows.apply(residency, axis=1)
print(rows[["model", "deployment", "level", "personal_data", "residency"]].to_string(index=False))
U.check(rows[rows.deployment.str.contains("Riyadh")].residency.eq("allowed").all(), "self-hosting in Riyadh is allowed for every candidate")

# %% [markdown]
# ## 3 · Cost per run (5')
#
# The workload, as Imad's IT colleague estimated it (synthetic, like the rates): late-delivery risk scores about
# 2,300 orders a month (every order once, when it is booked), well under an hour of CPU self-hosted; the weekly demand run forecasts 400
# series for 28 days, about 3 CPU hours or 0.5 GPU hours a week. Neither course run measured a real pretrained
# model's run time: the stand-ins answered (Days 1 and 2).
#
# ✏️ **Exercise** · Return the monthly cost in SAR of each option in `WORKLOAD`.

# %%
WORKLOAD = pd.DataFrame([
    {"task": "late-delivery risk (tables)", "option": "self-hosted CPU server (existing)", "hours_per_month": 0.5, "predictions_per_month": 2300},
    {"task": "late-delivery risk (tables)", "option": "hosted service, per 1,000 predictions", "hours_per_month": 0.0, "predictions_per_month": 2300},
    {"task": "daily demand (series)", "option": "self-hosted CPU server (existing)", "hours_per_month": 13.0, "predictions_per_month": 0},
    {"task": "daily demand (series)", "option": "self-hosted GPU server (to buy or lease)", "hours_per_month": 2.2, "predictions_per_month": 0},
])


def monthly_cost(workload: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    t = workload.merge(rates, on="option", how="left").fillna({"sar_per_hour": 0.0, "sar_per_1000": 0.0})
    return t.assign(sar_per_month=(t.hours_per_month * t.sar_per_hour + t.predictions_per_month / 1000 * t.sar_per_1000).round(0))
    # END SOLUTION

# %%
try:
    cost = monthly_cost(WORKLOAD, rates)
except NotImplementedError:
    t = WORKLOAD.merge(rates, on="option", how="left").fillna({"sar_per_hour": 0.0, "sar_per_1000": 0.0})
    cost = t.assign(sar_per_month=(t.hours_per_month * t.sar_per_hour + t.predictions_per_month / 1000 * t.sar_per_1000).round(0))
print(cost[["task", "option", "sar_per_month"]].to_string(index=False))
U.check((cost.sar_per_month >= 0).all() and cost.sar_per_month.max() < 1000, "every option costs under SAR 1,000 a month")

# %% [markdown]
# **Look at.** Running cost is small next to what Day 3 priced: about SAR 3,000 a week of regret between two
# forecasters for 20 products at one depot. Cost rarely decides; the licence, the data and the evidence do.

# %% [markdown]
# ## 4 · Adoption table and memo lines (4')
#
# The evidence of the course runs (stand-ins, not the pretrained models):

# %%
d12, d21 = U.reference("d1_2"), U.reference("d2_1")
lg, st = d12["table"][0], d12["table"][1]
diff = d12["diff"]
sql = {r["model"]: r["sql"] for r in d21["table"]}
print(f"Day 1 (2,000 orders): tuned LightGBM AUC {lg['auc']:.3f}, pretrained slot {st['auc']:.3f}; difference {diff[0]:+.3f} ({diff[1]:+.3f} to {diff[2]:+.3f}); answered by: {d12['answered_by']}")
print(f"Day 2 (400 series): scaled quantile loss LightGBM {sql['LightGBM (tuned)']:.3f}, zero-shot {sql['zero-shot']:.3f}; answered by: {d21['answered_by']}")

# %% [markdown]
# ✏️ **Exercise** · Complete `decision` with one row per candidate: `decision` (`"go"`, `"go with conditions"`,
# `"not now"` or `"no-go"`) and `memo`, one line each with the reason and the condition. Use the licence, residency
# and evidence above.

# %%
def decision() -> pd.DataFrame:
    # BEGIN SOLUTION
    return pd.DataFrame([
        {"model": "TabICLv2", "decision": "not now",
         "memo": "Licence and self-hosting allow it, but the course run cannot tell the pretrained slot from tuned LightGBM at any size; retest on Bustan's data with the real model."},
        {"model": "TabPFN-2.6", "decision": "no-go",
         "memo": "Non-commercial licence: weights and outputs may not steer Bustan's deliveries without a separate commercial licence."},
        {"model": "Chronos-2", "decision": "go with conditions",
         "memo": "Apache-2.0, self-hosted in Riyadh; as a member for holiday windows only, if its stacking weight pays for its running cost."},
        {"model": "TimesFM 2.5", "decision": "not now",
         "memo": "Apache-2.0 on its model page; not evaluated in this course, so no evidence either way."},
    ])
    # END SOLUTION

# %%
try:
    table = decision()
except NotImplementedError:
    table = pd.DataFrame(U.reference("d4_4")["decisions"])[["model", "decision", "memo"]]   # the course's own table
table = cands[["task", "model", "licence_ok"]].merge(table, on="model")
print(table.to_string(index=False))
U.check(set(table.decision) <= {"go", "go with conditions", "not now", "no-go"}, "every candidate has a decision")
U.check(table.set_index("model").loc["TabPFN-2.6", "decision"] == "no-go", "the blocked licence is a no-go")

# %% [markdown]
# ## 5 · Choose the capstone task (3')
#
# Your capstone writes the evaluation protocol and the adoption decision for **one** task, defended in 10 minutes
# with 10 minutes of questions. Choose the task where you can best show validation under shift.

# %%
CAPSTONE_TASK = "late-delivery risk (tables)"            # or "daily demand (series)"
U.check(CAPSTONE_TASK in DATASET, "a capstone task is chosen")
U.save_reference("d4_4", {"licences": cands[["model", "licence_ok", "licence_note"]].to_dict("records"),
                          "residency": rows[["model", "deployment", "residency"]].to_dict("records"),
                          "cost": cost[["task", "option", "sar_per_month"]].to_dict("records"),
                          "decisions": table.to_dict("records")})

# %% [markdown]
# ## ✱ Stretch · The protocol as a checklist
#
# Write ten lines another analyst could follow to rerun your evaluation on next quarter's data: the data cut,
# the validation scheme and why, the baseline and its budget, the metrics with intervals, the shift checks, the
# licence and residency checks, and the decision rule.

# %% [markdown]
# ## Checkpoint
#
# 1. A candidate's weights are Apache-2.0. Is the adoption question settled?
# 2. Why is Google Colab treated as outside the Kingdom?
# 3. Which of the four checks blocked a candidate outright in this lab?
#
# <details><summary>Answers</summary>
#
# 1. No: the data, its residency and the evidence of accuracy still decide.
# 2. The user does not choose where the machine runs, so Bustan cannot show the data stayed in the Kingdom.
# 3. The licence: TabPFN-2.6's outputs are for non-commercial use only.
#
# </details>
