# %% [markdown]
# # Lab 1.4 · Check a leaderboard claim
#
# Day 1 · Session 4 (Reading a leaderboard claim) · 25 min.
#
# Rafif Al-Moaibed forwards a vendor's slide to Ahlam Al-Kuwaileet: "Our pretrained model beats gradient
# boosting on 35 of 36 benchmark datasets." The vendor's table came with it. You recount the wins at equal
# tuning budgets, remove the datasets the model was pretrained or selected on, put an interval on the paired
# difference, and set the result beside Bustan's own learning curve.
#
# **AI policy: AI-tutor.**
#
# **Reference state.** `data/leaderboard_claim.csv` (synthetic: an invented vendor and invented datasets) and
# `data/reference/d1_3.json` (the learning-curve verdicts of Lab 1.3, supplied).
#
# **Tasks**
# - [ ] 1 · Read the claim and its table (4')
# - [ ] 2 · Equal budgets (6')
# - [ ] 3 · Drop overlap (5')
# - [ ] 4 · Paired difference with an interval (6')
# - [ ] 5 · Set it beside your own curve (4')
# - [ ] ✱ Stretch · Count how many datasets the claim would need to hold at equal budget

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

import numpy as np
import pandas as pd
import bustan as U

plt = U.style()

# %% [markdown]
# ## 1 · Read the claim and its table (4')
#
# Read the columns before counting anything: how many tuning trials did the boosted trees get as the claim
# ran them, and how many ensemble members did the pretrained model use?

# %%
lb = pd.read_csv(U.DATA / "leaderboard_claim.csv")
print(lb.head(8).to_string(index=False))
print("\nboosted-tree trials as claimed:", lb.gbm_trials_as_claimed.unique(), "| tuned column trials:", lb.gbm_trials_tuned.unique(),
      "| pretrained ensemble members:", lb.pretrained_ensemble_members.unique())
claimed = int((lb.pretrained_auc > lb.gbm_auc_as_claimed).sum())
print(f"wins as claimed: {claimed} of {len(lb)}")
U.check(claimed == 35, "the table reproduces the claim: 35 of 36")

# %% [markdown]
# ## 2 · Equal budgets (6')
#
# The claim compared a pretrained model (with its own built-in ensemble) to boosted trees left at their
# defaults. The table also holds the boosted trees after 100 tuning trials.
#
# ✏️ **Exercise** · Return the number of datasets where the pretrained model beats the **tuned** trees.

# %%
def wins_equal_budget(lb: pd.DataFrame) -> int:
    # BEGIN SOLUTION
    return int((lb.pretrained_auc > lb.gbm_auc_tuned).sum())
    # END SOLUTION

# %%
try:
    equal = wins_equal_budget(lb)
except NotImplementedError:
    equal = int((lb.pretrained_auc > lb.gbm_auc_tuned).sum())
print(f"wins at equal budget: {equal} of {len(lb)}")
U.check(equal < claimed, "tuning the baseline removes wins")

# %% [markdown]
# ## 3 · Drop overlap (5')
#
# `in_pretraining_or_selection` marks datasets the vendor used to pretrain or to choose the model's settings.
# A model scored on data it was selected on is scored on its homework.
#
# ✏️ **Exercise** · Return the rows without overlap.

# %%
def without_overlap(lb: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    return lb[lb.in_pretraining_or_selection == 0].reset_index(drop=True)
    # END SOLUTION

# %%
try:
    clean = without_overlap(lb)
except NotImplementedError:
    clean = lb[lb.in_pretraining_or_selection == 0].reset_index(drop=True)
clean_wins = int((clean.pretrained_auc > clean.gbm_auc_tuned).sum())
print(f"{len(lb) - len(clean)} datasets dropped; wins at equal budget without overlap: {clean_wins} of {len(clean)}")
U.check(clean.in_pretraining_or_selection.sum() == 0 and len(clean) < len(lb), "no overlapping dataset left")

# %% [markdown]
# ## 4 · Paired difference with an interval (6')
#
# Counting wins treats a 0.001 win like a 0.05 win. The paired difference per dataset, with a bootstrap
# interval over datasets, says how big the advantage is and whether it is distinguishable from zero.
#
# ✏️ **Exercise** · Return a table with one row per slice (`all`, `under 3,000 rows`, `3,000 rows or more`) of
# the clean datasets and the columns `datasets`, `mean_diff`, `lo`, `hi` (pretrained minus tuned, `U.bootstrap_ci`).

# %%
def paired_table(clean: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    d = clean.pretrained_auc - clean.gbm_auc_tuned
    rows = []
    for name, sel in (("all", np.ones(len(clean), bool)), ("under 3,000 rows", clean.rows < 3000),
                      ("3,000 rows or more", clean.rows >= 3000)):
        m, lo, hi = U.bootstrap_ci(d[sel])
        rows.append({"slice": name, "datasets": int(sel.sum()), "mean_diff": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
    # END SOLUTION

# %%
try:
    paired = paired_table(clean)
except NotImplementedError:
    d = clean.pretrained_auc - clean.gbm_auc_tuned
    paired = pd.DataFrame([{"slice": n, "datasets": int(s.sum()), **dict(zip(["mean_diff", "lo", "hi"], U.bootstrap_ci(d[s])))}
                           for n, s in (("all", np.ones(len(clean), bool)), ("under 3,000 rows", clean.rows < 3000),
                                        ("3,000 rows or more", clean.rows >= 3000))])
print(paired.round(4).to_string(index=False))
U.check(paired.lo.iloc[0] < 0 < paired.hi.iloc[0], "over all clean datasets, the interval of the difference contains zero")
U.save_reference("d1_4", {"claimed": claimed, "equal": equal, "clean_wins": clean_wins, "clean_n": len(clean),
                          "paired": paired.to_dict("records")})

# %% [markdown]
# ## 5 · Set it beside your own curve (4')
#
# The vendor's table is someone else's data. Bustan's own evidence is the learning curve of Lab 1.3.
# Compare the sign of the small-data and large-data slices with the verdicts per size.

# %%
own = pd.DataFrame(U.reference("d1_3")["diffs"])
print("Bustan's learning curve (pretrained minus baseline), answered by:", U.reference("d1_3")["answered_by"])
print(own.round(4).to_string(index=False))

# %% [markdown]
# **Write the reply to Rafif in three lines:** what the claim compared, what remains at equal budget and
# without overlap, and what Bustan's own curve says.

# %% [markdown]
# ## ✱ Stretch · How strong would the claim need to be?
#
# With the clean datasets' spread of differences, how many datasets would a mean advantage of 0.005 AUC need
# for its 95% interval to exclude zero? (Hint: the interval half-width shrinks with the square root of the count.)

# %% [markdown]
# ## Checkpoint
#
# 1. Which single column of the table voids the "35 of 36" most?
# 2. A dataset is public and the vendor's model card lists it among its validation suites. Keep it?
# 3. The clean small-data slice favours the pretrained model. What would make you act on it at Bustan?
#
# <details><summary>Answers</summary>
#
# 1. `gbm_trials_as_claimed`: zero tuning trials against a model that ensembles eight passes.
# 2. No: it was used to select the model, so it cannot also judge it.
# 3. The same sign on Bustan's own curve at Bustan's sizes, with an interval that excludes zero.
#
# </details>
#
# **Carry forward.** The recount is in `data/reference/d1_4.json`.
