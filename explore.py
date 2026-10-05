import pandas as pd

data = pd.read_csv("creditcard.csv")

print(data.shape) 
print(data.head())
print(data["Class"].value_counts())
print(data.groupby("Class")["Amount"].describe())