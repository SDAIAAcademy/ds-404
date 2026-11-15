# %% [markdown]
# # Setup check
#
# Run this once before Day 1, on the machine you will use (a laptop with uv, or Google Colab). It checks the
# Python packages, reads one file of each kind, solves a tiny optimisation, and says whether the pretrained
# models can run here or the stand-in will answer.
#
# **Pre-work.** Before Day 1, complete the three hours of guided pre-work on quantile loss, newsvendor logic
# and linear programmes, and its entry quiz (the practice questions tagged "pre-work" on the course page).

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

# %%
import importlib

missing = []
for pkg in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "matplotlib", "pulp", "highspy"]:
    try:
        m = importlib.import_module(pkg)
        print(f"✓ {pkg} {getattr(m, '__version__', '')}")
    except ImportError:
        missing.append(pkg)
        print(f"✗ {pkg} is missing")
print("Python", sys.version.split()[0], "| Colab" if IN_COLAB else "| local")

# %%
import bustan as U

o = U.orders()
Y = U.demand()
print(f"orders: {len(o):,} rows; demand: {Y.shape[1]} series x {len(Y):,} days; calendar: {len(U.calendar()):,} days")
U.check(len(o) == 80000 and Y.shape[1] == 400, "the case data is in place")

# %%
import pulp

prob = pulp.LpProblem("tiny", pulp.LpMinimize)
x = prob.add_variable("trucks", lowBound=0, cat="Integer")      # PuLP 4: variables belong to a problem
prob += 2400 * x
prob += 60 * x >= 130
prob.solve(pulp.HiGHS(msg=False))
U.check(pulp.value(x) == 3, "HiGHS solves an integer programme: 3 trucks carry 130 cubic metres")

# %%
_, info_tab = U.tabular_model()
print("tables:", info_tab["model"])
try:
    import chronos  # noqa: F401
    print("series: Chronos-2 can run here")
except ImportError:
    print("series: the stand-in will answer (install chronos-forecasting in Colab for the real model)")
print("\nAll set." if not missing else f"\nInstall first: {missing}")
