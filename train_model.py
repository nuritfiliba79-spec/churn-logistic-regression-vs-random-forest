"""Train Logistic Regression, Random Forest, XGBoost, LightGBM and CatBoost on churn data and export results to data.js.

The final models use the hyperparameters found by tune_models.py (best_params.json).
The original (default) models are also scored, to show the effect of tuning.
"""
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
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "churn_modelling.csv")

# ---- Step 1-3: data processing BEFORE the split ----
# These steps do not learn anything from the data (each row is handled on its own),
# so doing them before the split cannot leak information from the test set.
assert df.isna().sum().sum() == 0 and df.duplicated().sum() == 0   # no missing / duplicate rows
cat_cols = ["Geography", "Gender"]
num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary"]
data = df.drop(columns=["CustomerId", "Surname"])                  # identifiers carry no signal
data = pd.get_dummies(data, columns=cat_cols, drop_first=True, dtype=int)  # one-hot -> 0/1 columns
X = data.drop(columns=["Exited"])
y = data["Exited"]

# ---- Step 4: 80% train / 20% test, random and stratified ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---- Step 5: scaling AFTER the split ----
# StandardScaler learns a mean and std, so it lives inside the Pipeline:
# it is fitted on the training data only (and on the training folds during CV).
def prep():
    return ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="passthrough")


def pipe(clf):
    return Pipeline([("prep", prep()), ("clf", clf)])


tuning = json.loads((HERE / "best_params.json").read_text(encoding="utf-8"))
bp = {k: v["best_params"] for k, v in tuning["models"].items()}

# Original parameters (before tuning)
default_models = {
    "lr": pipe(LogisticRegression(max_iter=1000)),
    "rf": pipe(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    "xgb": pipe(XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42)),
    "lgbm": pipe(LGBMClassifier(random_state=42, verbose=-1, n_jobs=1)),
    "cat": pipe(CatBoostClassifier(random_seed=42, verbose=0, thread_count=1, allow_writing_files=False)),
}
# Final models with the best parameters from GridSearchCV
models = {
    "lr": pipe(LogisticRegression(max_iter=2000, solver="liblinear", **bp["lr"])),
    "rf": pipe(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, **bp["rf"])),
    "xgb": pipe(XGBClassifier(random_state=42, n_jobs=1, **bp["xgb"])),  # n_jobs=1: same result as in the CV search
    "lgbm": pipe(LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, **bp["lgbm"])),
    "cat": pipe(CatBoostClassifier(random_seed=42, verbose=0, thread_count=1, allow_writing_files=False, **bp["cat"])),
}


def test_scores(m):
    pred, p = m.predict(X_test), m.predict_proba(X_test)[:, 1]
    return {"recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
            "f1": f1_score(y_test, pred), "auc": roc_auc_score(y_test, p),
            "train_accuracy": m.score(X_train, y_train), "accuracy": accuracy_score(y_test, pred)}


out = {"n_train": len(X_train), "n_test": len(X_test), "churn_rate": float(y.mean()),
       "cv_folds": tuning["cv_folds"], "models": {}}

# Facts about each step of the process, shown in the "process" tab
out["pipeline"] = {
    "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
    "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
    "dropped": ["CustomerId", "Surname"], "target": "Exited",
    "num_cols": num_cols, "cat_cols": {c: sorted(df[c].unique().tolist()) for c in cat_cols},
    "n_features_in": len(num_cols) + len(cat_cols), "features_out": list(X.columns),
    "train_churn": float(y_train.mean()), "test_churn": float(y_test.mean()),
    "train_churners": int(y_train.sum()), "test_churners": int(y_test.sum()),
    "overlap_ids": int(len(set(df.loc[X_train.index, "CustomerId"]) & set(df.loc[X_test.index, "CustomerId"]))),
}
for key, m in models.items():
    default_test = test_scores(default_models[key].fit(X_train, y_train))
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
    # LR: |standardized coefficient|, RF: Gini, XGBoost/LightGBM: gain, CatBoost: PredictionValuesChange
    if key == "lr":
        raw = np.abs(clf.coef_[0])
    elif key == "lgbm":
        raw = clf.booster_.feature_importance(importance_type="gain")
    elif key == "cat":
        raw = clf.get_feature_importance()
    else:
        raw = clf.feature_importances_
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
        "default_test": default_test,
        "tuning": tuning["models"][key],
    }
    print(key, "tuned  ", {k: round(v, 3) for k, v in out["models"][key]["metrics"].items()})
    print(key, "default", {k: round(v, 3) for k, v in default_test.items()})

(HERE / "data.js").write_text("window.DASH = " + json.dumps(out) + ";\n", encoding="utf-8")
print("wrote data.js")
