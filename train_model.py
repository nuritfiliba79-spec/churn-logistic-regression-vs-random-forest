"""Train Logistic Regression, Random Forest and XGBoost on churn data and export results to data.js."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "churn_modelling.csv")

# Identifiers carry no predictive signal
X = df.drop(columns=["CustomerId", "Surname", "Exited"])
y = df["Exited"]
cat_cols = ["Geography", "Gender"]
num_cols = [c for c in X.columns if c not in cat_cols]

# 80% train / 20% test, random and stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols),
])
models = {
    "lr": Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))]),
    "rf": Pipeline([("prep", prep), ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]),
    "xgb": Pipeline([("prep", prep), ("clf", XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42))]),
}

out = {"n_train": len(X_train), "n_test": len(X_test), "churn_rate": float(y.mean()), "models": {}}
for key, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    pred = m.predict(X_test)
    resid = y_test.to_numpy() - p
    fpr, tpr, _ = roc_curve(y_test, p)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    # Actual vs predicted: split test set into 10 bins by predicted probability
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    calib = (pd.DataFrame({"p": p, "y": y_test.to_numpy(), "b": bins})
             .groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")))

    hist, edges = np.histogram(resid, bins=20, range=(-1, 1))

    clf = m.named_steps["clf"]
    names = [n.split("__")[1] for n in m.named_steps["prep"].get_feature_names_out()]
    # RF: Gini importance, XGBoost: gain importance, LR: |standardized coefficient|
    raw = np.abs(clf.coef_[0]) if key == "lr" else clf.feature_importances_
    share = np.asarray(raw, dtype=float) / np.sum(raw)

    out["models"][key] = {
        "metrics": {
            "recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
            "f1": f1_score(y_test, pred), "auc": roc_auc_score(y_test, p),
            "accuracy": accuracy_score(y_test, pred), "train_accuracy": m.score(X_train, y_train),
            "r2": r2_score(y_test, p), "mae": mean_absolute_error(y_test, p),
        },
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": [[round(a, 4), round(b, 4)] for a, b in zip(fpr, tpr)],
        "calibration": [{"pred": r.pred, "actual": r.actual, "n": int(r.n)} for r in calib.itertuples()],
        "residual_hist": {"edges": edges.round(2).tolist(), "counts": hist.tolist()},
        "importance": dict(zip(names, share.round(4).tolist())),
    }
    print(key, {k: round(v, 3) for k, v in out["models"][key]["metrics"].items()})

(HERE / "data.js").write_text("window.DASH = " + json.dumps(out) + ";\n", encoding="utf-8")
print("wrote data.js")
