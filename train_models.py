"""Train XGBoost and LightGBM on the UCI Chronic Kidney Disease dataset.

Download kidney_disease.csv (UCI CKD dataset) into this folder, then run:
    pip install pandas scikit-learn xgboost lightgbm joblib
    python train_models.py
"""
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

df = pd.read_csv("kidney_disease.csv")
y = (df.pop("classification").astype(str).str.strip() == "ckd").astype(int)
df = df.drop(columns="id", errors="ignore")
for c in df.select_dtypes("object"):
    df[c] = df[c].astype(str).str.strip().replace({"?": None, "nan": None})
X = pd.get_dummies(df)
X = X.fillna(X.median(numeric_only=True))

models = {
    "xgb": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="logloss"),
    "lgbm": LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, verbose=-1),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, model in models.items():
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"{name}: cross-validated ROC-AUC = {auc:.3f}")
    joblib.dump(model.fit(X, y), f"{name}.joblib")
joblib.dump(list(X.columns), "feature_columns.joblib")
