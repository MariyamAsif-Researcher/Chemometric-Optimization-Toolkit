import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ===========================================
# Load Dataset
# ===========================================

data = pd.read_csv("data/GCC_DEMO.csv")

# ===========================================
# Response Variables
# ===========================================

responses = [
    "TPC",
    "TFC",
    "DPPH",
    "ABTS",
    "MCA"
]

all_results = []

print("="*80)
print("DESIGN EXPERT STYLE QUADRATIC ANOVA")
print("="*80)

for response in responses:

    print("\n")
    print("="*70)
    print(f"Response Variable : {response}")
    print("="*70)

    formula = f"""
    {response} ~
    DES +
    Power +
    Time +
    DES:Power +
    DES:Time +
    Power:Time +
    I(DES**2) +
    I(Power**2) +
    I(Time**2)
    """

    model = ols(formula, data=data).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    anova_table.to_csv(f"results/{response}_ANOVA.csv")

    all_results.append(model.rsquared)

print("\nFinished.")