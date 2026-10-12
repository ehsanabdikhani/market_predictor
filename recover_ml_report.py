import os
import glob
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

RESULT_ROOT = "train_result_fib_LTF"
MODEL_ROOT = "trained_models"

TARGET = "Movement_Signal"

TEST_SIZE = 0.20
CV_SPLITS = 3
RANDOM_STATE = 42
MIN_SAMPLES = 100

# Output files: same names/columns as the original script
RESULTS_FILE = "ml_results.csv"
BEST_CSV_FILE = "ml_best_per_csv.csv"
BEST_PAIR_FILE = "ml_best_per_pair.csv"


# ============================================================
# PREPARE DATA EXACTLY LIKE THE ORIGINAL SCRIPT
# ============================================================

def prepare_data(filepath):
    df = pd.read_csv(filepath)

    df = df.dropna(axis=1, how="all")

    if TARGET not in df.columns:
        return None

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    if len(df) < MIN_SAMPLES:
        return None

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Date is not a feature
    if "Date" in X.columns:
        X = X.drop(columns=["Date"])

    # Remove completely empty columns
    empty_columns = [
        c for c in X.columns
        if X[c].isna().all()
    ]

    if empty_columns:
        X = X.drop(columns=empty_columns)

    # IMPORTANT:
    # This is the exact same split as the training script because
    # the original script used random_state=42 and stratify=y.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return df, X_train, X_test, y_train, y_test


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(model, X_test, y_test):
    prediction = model.predict(X_test)

    result = {}

    result["accuracy"] = accuracy_score(
        y_test,
        prediction
    )

    result["balanced_accuracy"] = balanced_accuracy_score(
        y_test,
        prediction
    )

    result["precision"] = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    result["recall"] = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    result["f1"] = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    try:
        probabilities = model.predict_proba(X_test)

        # For a Pipeline, classes_ is normally exposed by the final
        # estimator through Pipeline.classes_.
        classes = model.classes_

        if len(classes) == 2:
            result["roc_auc"] = roc_auc_score(
                y_test,
                probabilities[:, 1]
            )
        else:
            result["roc_auc"] = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="weighted"
            )

    except Exception:
        result["roc_auc"] = np.nan

    return result, prediction


# ============================================================
# RECOVER BEST PARAMETERS FROM SAVED PIPELINE
# ============================================================

def get_best_params(saved_model):
    """
    The original script saved only best_estimator_, not the complete
    RandomizedSearchCV object.

    Therefore search.best_params_ itself is not available anymore.
    We reconstruct the selected model parameters from the saved
    fitted pipeline and return them using the same model__... naming
    convention as the original report.
    """

    try:
        model = saved_model.named_steps["model"]
        params = model.get_params()

        # Keep only hyperparameters that were searched in the
        # original script.
        searched_names = {
            "LogisticRegression": [
                "C", "solver"
            ],
            "RandomForest": [
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features"
            ],
            "ExtraTrees": [
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features"
            ],
            "GradientBoosting": [
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_samples_leaf"
            ],
            "HistGradientBoosting": [
                "max_iter",
                "learning_rate",
                "max_leaf_nodes",
                "min_samples_leaf"
            ],
            "SVM": [
                "C", "gamma", "kernel"
            ],
            "KNN": [
                "n_neighbors", "weights", "p"
            ]
        }

        # model name is supplied separately in the caller, so this
        # function only returns all relevant estimator parameters.
        return params

    except Exception:
        return {}


def format_best_params(saved_model, model_name):
    params = get_best_params(saved_model)

    searched_names = {
        "LogisticRegression": [
            "C", "solver"
        ],
        "RandomForest": [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features"
        ],
        "ExtraTrees": [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features"
        ],
        "GradientBoosting": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_leaf"
        ],
        "HistGradientBoosting": [
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "min_samples_leaf"
        ],
        "SVM": [
            "C", "gamma", "kernel"
        ],
        "KNN": [
            "n_neighbors", "weights", "p"
        ]
    }

    names = searched_names.get(model_name, [])

    best_params = {}

    for name in names:
        if name in params:
            best_params["model__" + name] = params[name]

    return str(best_params)


# ============================================================
# RECOMPUTE CV SCORE FOR THE SAVED MODEL
# ============================================================

def recover_cv_balanced_accuracy(model, X_train, y_train):
    """
    The original code stored search.best_score_, but did not save the
    RandomizedSearchCV object.

    We therefore evaluate the already-selected estimator using the
    exact same 3-fold StratifiedKFold definition.

    With the original fixed random states and unchanged data, this
    reproduces the CV score for the selected hyperparameters.
    """

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    try:
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1
        )

        return float(np.mean(scores))

    except Exception as e:
        print("WARNING: Could not recover CV score:", e)
        return np.nan


# ============================================================
# PROCESS ONE CSV
# ============================================================

def process_file(filepath):
    print("\n")
    print("=" * 100)
    print("FILE:", filepath)
    print("=" * 100)

    pair = os.path.basename(
        os.path.dirname(filepath)
    )

    filename = os.path.basename(filepath)

    prepared = prepare_data(filepath)

    if prepared is None:
        print("SKIPPED: target missing or too few samples.")
        return []

    df, X_train, X_test, y_train, y_test = prepared

    print("Samples:", len(df))
    print(
        "Train:",
        len(X_train),
        "| Test:",
        len(X_test)
    )

    print("\nTarget distribution:")
    print(df[TARGET].value_counts())

    all_results = []

    # --------------------------------------------------------
    # Look for every model that should exist for this CSV.
    # --------------------------------------------------------

    model_names = [
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "HistGradientBoosting",
        "SVM",
        "KNN"
    ]

    model_dir = os.path.join(
        MODEL_ROOT,
        pair
    )

    for model_name in model_names:

        model_filename = (
            filename.replace(".csv", "")
            + "_"
            + model_name
            + ".joblib"
        )

        model_path = os.path.join(
            model_dir,
            model_filename
        )

        if not os.path.exists(model_path):
            print(
                "\nMODEL FILE NOT FOUND:",
                model_path
            )
            continue

        print("\n")
        print("-" * 80)
        print("RECOVERING MODEL:", model_name)
        print("-" * 80)
        print("MODEL:", model_path)

        try:
            best_model = joblib.load(model_path)

            # ------------------------------------------------
            # TEST METRICS
            # ------------------------------------------------

            metrics, prediction = calculate_metrics(
                best_model,
                X_test,
                y_test
            )

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            cm = confusion_matrix(
                y_test,
                prediction,
                labels=best_model.classes_
            )

            # ------------------------------------------------
            # RECOVER CV SCORE
            # ------------------------------------------------

            cv_score = recover_cv_balanced_accuracy(
                best_model,
                X_train,
                y_train
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result = {
                "pair": pair,
                "file": filename,
                "model": model_name,

                "samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test),

                "cv_balanced_accuracy": cv_score,

                "test_accuracy": metrics["accuracy"],

                "test_balanced_accuracy":
                    metrics["balanced_accuracy"],

                "test_precision":
                    metrics["precision"],

                "test_recall":
                    metrics["recall"],

                "test_f1":
                    metrics["f1"],

                "test_roc_auc":
                    metrics["roc_auc"],

                "best_params":
                    format_best_params(
                        best_model,
                        model_name
                    ),

                "confusion_matrix":
                    str(cm),

                "model_path":
                    model_path
            }

            all_results.append(result)

            # ------------------------------------------------
            # PRINT RESULTS
            # ------------------------------------------------

            print(
                "\nCV balanced accuracy:",
                round(cv_score, 4)
                if not np.isnan(cv_score)
                else "N/A"
            )

            print(
                "TEST accuracy:",
                round(metrics["accuracy"], 4)
            )

            print(
                "TEST balanced accuracy:",
                round(
                    metrics["balanced_accuracy"],
                    4
                )
            )

            print(
                "TEST precision:",
                round(
                    metrics["precision"],
                    4
                )
            )

            print(
                "TEST recall:",
                round(
                    metrics["recall"],
                    4
                )
            )

            print(
                "TEST F1:",
                round(
                    metrics["f1"],
                    4
                )
            )

            if np.isnan(metrics["roc_auc"]):
                print("TEST ROC-AUC: N/A")
            else:
                print(
                    "TEST ROC-AUC:",
                    round(
                        metrics["roc_auc"],
                        4
                    )
                )

            print("\nBEST PARAMETERS:")
            print(
                format_best_params(
                    best_model,
                    model_name
                )
            )

        except Exception as e:
            print(
                "ERROR RECOVERING:",
                model_name
            )
            print(e)

    return all_results


# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = glob.glob(
    os.path.join(
        RESULT_ROOT,
        "*",
        "*.csv"
    )
)

print(
    "Found",
    len(csv_files),
    "CSV files"
)


# ============================================================
# RECOVER ALL RESULTS FROM SAVED MODELS
# ============================================================

all_results = []

for filepath in csv_files:

    results = process_file(
        filepath
    )

    all_results.extend(
        results
    )


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

if len(results_df) > 0:

    # ========================================================
    # SORT
    # ========================================================

    results_df = (
        results_df
        .sort_values(
            [
                "test_balanced_accuracy",
                "test_f1",
                "test_accuracy"
            ],
            ascending=False
        )
    )

    # ========================================================
    # SAVE ALL RESULTS
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ========================================================
    # PRINT ALL
    # ========================================================

    print("\n")
    print("=" * 130)
    print("ALL MODEL RESULTS")
    print("=" * 130)

    print(
        results_df[
            [
                "pair",
                "file",
                "model",
                "cv_balanced_accuracy",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST MODEL PER CSV
    # ========================================================

    best_per_csv = (
        results_df
        .sort_values(
            "test_balanced_accuracy",
            ascending=False
        )
        .groupby(
            [
                "pair",
                "file"
            ],
            as_index=False
        )
        .first()
    )

    print("\n")
    print("=" * 130)
    print("BEST MODEL PER CSV")
    print("=" * 130)

    print(
        best_per_csv[
            [
                "pair",
                "file",
                "model",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "model_path"
            ]
        ].to_string(
            index=False
        )
    )

    best_per_csv.to_csv(
        BEST_CSV_FILE,
        index=False
    )

    # ========================================================
    # BEST MODEL PER PAIR
    # ========================================================

    best_per_pair = (
        results_df
        .sort_values(
            "test_balanced_accuracy",
            ascending=False
        )
        .groupby(
            "pair",
            as_index=False
        )
        .first()
    )

    print("\n")
    print("=" * 130)
    print("BEST MODEL PER PAIR")
    print("=" * 130)

    print(
        best_per_pair[
            [
                "pair",
                "file",
                "model",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "model_path"
            ]
        ].to_string(
            index=False
        )
    )

    best_per_pair.to_csv(
        BEST_PAIR_FILE,
        index=False
    )

    print("\n")
    print("=" * 130)
    print("REPORT RECOVERY COMPLETE")
    print("=" * 130)
    print("Saved:", RESULTS_FILE)
    print("Saved:", BEST_CSV_FILE)
    print("Saved:", BEST_PAIR_FILE)

else:

    print("\nNo saved models could be recovered.")
