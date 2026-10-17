"""Hyperparameter search: GridSearchCV with stratified 5-fold CV on the training set only.

Writes best_params.json, which train_model.py uses to build the final models.
Run `python tune_models.py lr` to re-tune only some models and keep the others.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
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

# ---- Step 4: same split as train_model.py; the test set is never used here ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
neg_pos = float((y_train == 0).sum() / (y_train == 1).sum())  # ~3.9


# ---- Step 5: scaling AFTER the split ----
# StandardScaler learns a mean and std, so it lives inside the Pipeline:
# it is fitted on the training data only (and on the training folds during CV).
def prep():
    return ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="passthrough")


MODELS = {
    "lr": {
        "default": LogisticRegression(max_iter=1000),
        "grid": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__l1_ratio": [0, 1],  # 0 = L2 (ridge), 1 = L1 (lasso)
            "clf__class_weight": [None, "balanced"],
        },
        "base": LogisticRegression(max_iter=2000, solver="liblinear"),
    },
    "rf": {
        "default": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
        "grid": {
            "clf__max_depth": [None, 6, 10, 14],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced_subsample"],
        },
        "base": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
    },
    "xgb": {
        "default": XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42, n_jobs=1),
        "grid": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__scale_pos_weight": [1, round(neg_pos, 2)],
        },
        "base": XGBClassifier(random_state=42, n_jobs=1),
    },
    "lgbm": {
        "default": LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
        "grid": {
            "clf__n_estimators": [100, 200, 300],
            "clf__num_leaves": [15, 31, 63],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__min_child_samples": [20, 40],
            "clf__class_weight": [None, "balanced"],
        },
        "base": LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
    },
    "cat": {
        "default": CatBoostClassifier(random_seed=42, verbose=0, thread_count=1, allow_writing_files=False),
        "grid": {
            "clf__iterations": [300, 600],
            "clf__depth": [4, 6, 8],
            "clf__learning_rate": [0.03, 0.1],
            "clf__auto_class_weights": [None, "Balanced"],
        },
        "base": CatBoostClassifier(random_seed=42, verbose=0, thread_count=1, allow_writing_files=False),
    },
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc"}


def summary(means, stds):
    return {k: {"mean": float(means[k]), "std": float(stds[k])} for k in SCORING}


only = sys.argv[1:] or list(MODELS)
path = HERE / "best_params.json"
out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"models": {}}
out.update({"cv_folds": 5, "refit_metric": "f1", "neg_pos_ratio": neg_pos})
for key, spec in MODELS.items():
    if key not in only:
        continue
    t0 = time.time()
    # Baseline: the original parameters, scored with the same folds
    base = cross_validate(Pipeline([("prep", prep()), ("clf", spec["default"])]), X_train, y_train,
                          cv=cv, scoring=SCORING, n_jobs=-1)
    base_sum = summary({k: base["test_" + k].mean() for k in SCORING}, {k: base["test_" + k].std() for k in SCORING})

    gs = GridSearchCV(Pipeline([("prep", prep()), ("clf", spec["base"])]), spec["grid"],
                      scoring=SCORING, refit="f1", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    i = gs.best_index_
    r = gs.cv_results_
    best_sum = summary({k: r["mean_test_" + k][i] for k in SCORING}, {k: r["std_test_" + k][i] for k in SCORING})
    best = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    n_combos = len(r["params"])

    out["models"][key] = {
        "grid": {k.replace("clf__", ""): v for k, v in spec["grid"].items()},
        "n_combos": n_combos,
        "n_fits": n_combos * 5,
        "best_params": best,
        "cv_default": base_sum,
        "cv_best": best_sum,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"{key}: {n_combos} combos, {time.time() - t0:.0f}s  best={best}")
    print(f"   CV F1 default {base_sum['f1']['mean']:.3f} -> tuned {best_sum['f1']['mean']:.3f} (±{best_sum['f1']['std']:.3f})")

path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print("wrote best_params.json")
