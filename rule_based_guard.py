import pandas as pd

data = pd.read_csv("creditcard.csv")

def rule_based_check(amount):
    if amount < 10:
        return 1  # flag as suspicious
    else:
        return 0  # looks normal

data["rule_prediction"] = data["Amount"].apply(rule_based_check)

print(data["rule_prediction"].value_counts())

comparison = pd.crosstab(data["Class"], data["rule_prediction"])
print(comparison)
print(data.groupby("Class")["Time"].describe())