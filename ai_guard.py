import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

print("Training done!")

predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix

comparison = confusion_matrix(y_test, predictions)
print(comparison)

import shap

explainer = shap.TreeExplainer(model)

fraud_rows = X_test[(y_test == 1) & (predictions == 1)]
one_fraud = fraud_rows.iloc[[0]]

shap_values = explainer.shap_values(one_fraud)

explanation = pd.Series(shap_values[0], index=X_test.columns)
print(explanation.sort_values(key=abs, ascending=False).head(5))