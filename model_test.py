import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_ROOT = "trained_models_RSI_version"

UNSEEN_ROOT = "result_new"

TARGET = "Movement_Signal"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(model, X, y):

    prediction = model.predict(X)

    result = {}

    result["accuracy"] = accuracy_score(
        y,
        prediction
    )

    result["balanced_accuracy"] = (
        balanced_accuracy_score(
            y,
            prediction
        )
    )

    result["precision"] = precision_score(
        y,
        prediction,
        average="weighted",
        zero_division=0
    )

    result["recall"] = recall_score(
        y,
        prediction,
        average="weighted",
        zero_division=0
    )

    result["f1"] = f1_score(
        y,
        prediction,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    try:

        probabilities = model.predict_proba(X)

        if len(model.classes_) == 2:

            result["roc_auc"] = roc_auc_score(
                y,
                probabilities[:, 1]
            )

        else:

            result["roc_auc"] = roc_auc_score(
                y,
                probabilities,
                multi_class="ovr",
                average="weighted"
            )

    except Exception:

        result["roc_auc"] = np.nan

    return result, prediction


# ============================================================
# FIND ALL TRAINED MODELS
# ============================================================

model_files = glob.glob(
    os.path.join(
        MODEL_ROOT,
        "*",
        "*.joblib"
    )
)

print(
    "Found",
    len(model_files),
    "trained models"
)


# ============================================================
# TEST MODELS
# ============================================================

all_results = []


for model_path in model_files:

    # --------------------------------------------------------
    # Get pair
    # --------------------------------------------------------

    pair = os.path.basename(
        os.path.dirname(model_path)
    )

    model_filename = os.path.basename(
        model_path
    )

    print("\n")
    print("=" * 100)

    print(
        "MODEL:",
        model_filename
    )

    print(
        "PAIR:",
        pair
    )

    print("=" * 100)


    # --------------------------------------------------------
    # Remove model name from filename
    #
    # Example:
    #
    # EURUSD_10_RandomForest.joblib
    #
    # becomes:
    #
    # EURUSD_10.csv
    # --------------------------------------------------------

    csv_name = model_filename.replace(
        ".joblib",
        ""
    )

    model_names = [
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "HistGradientBoosting",
        "SVM",
        "KNN"
    ]

    for model_name in model_names:

        suffix = "_" + model_name

        if csv_name.endswith(suffix):

            csv_name = csv_name[
                :-len(suffix)
            ]

            break


    csv_name += ".csv"


    # --------------------------------------------------------
    # Find unseen CSV
    # --------------------------------------------------------

    unseen_path = os.path.join(
        UNSEEN_ROOT,
        pair,
        csv_name
    )


    if not os.path.exists(
        unseen_path
    ):

        print(
            "UNSEEN CSV NOT FOUND:",
            unseen_path
        )

        continue


    # --------------------------------------------------------
    # Load unseen data
    # --------------------------------------------------------

    df = pd.read_csv(
        unseen_path
    )


    if TARGET not in df.columns:

        print(
            "TARGET NOT FOUND:",
            unseen_path
        )

        continue


    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------

    df = df.dropna(
        subset=[TARGET]
    ).reset_index(
        drop=True
    )


    if len(df) == 0:

        print(
            "No valid rows"
        )

        continue


    # --------------------------------------------------------
    # Features / target
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]


    # --------------------------------------------------------
    # Date is not a feature
    # --------------------------------------------------------

    if "Date" in X.columns:

        X = X.drop(
            columns=["Date"]
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = joblib.load(
            model_path
        )

    except Exception as e:

        print(
            "MODEL LOAD ERROR:",
            e
        )

        continue


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    try:

        metrics, prediction = (
            calculate_metrics(
                model,
                X,
                y
            )
        )

    except Exception as e:

        print(
            "PREDICTION ERROR:",
            e
        )

        continue


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    try:

        cm = confusion_matrix(
            y,
            prediction,
            labels=model.classes_
        )

    except:

        cm = None


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "pair":
            pair,

        "csv":
            csv_name,

        "model":
            model_name,

        "samples":
            len(df),

        "accuracy":
            metrics["accuracy"],

        "balanced_accuracy":
            metrics["balanced_accuracy"],

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1"],

        "roc_auc":
            metrics["roc_auc"],

        "confusion_matrix":
            str(cm),

        "model_path":
            model_path
    }


    all_results.append(
        result
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "Samples:",
        len(df)
    )

    print(
        "Accuracy:",
        round(
            metrics["accuracy"],
            4
        )
    )

    print(
        "Balanced accuracy:",
        round(
            metrics["balanced_accuracy"],
            4
        )
    )

    print(
        "Precision:",
        round(
            metrics["precision"],
            4
        )
    )

    print(
        "Recall:",
        round(
            metrics["recall"],
            4
        )
    )

    print(
        "F1:",
        round(
            metrics["f1"],
            4
        )
    )

    if np.isnan(
        metrics["roc_auc"]
    ):

        print(
            "ROC-AUC: N/A"
        )

    else:

        print(
            "ROC-AUC:",
            round(
                metrics["roc_auc"],
                4
            )
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


if len(results_df) > 0:

    results_df = (
        results_df
        .sort_values(
            [
                "balanced_accuracy",
                "f1",
                "accuracy"
            ],
            ascending=False
        )
    )


    results_df.to_csv(
        "unseen_results_2026.csv",
        index=False
    )


    # ========================================================
    # ALL RESULTS
    # ========================================================

    print("\n")
    print("=" * 130)

    print(
        "ALL UNSEEN RESULTS"
    )

    print("=" * 130)


    print(

        results_df[
            [
                "pair",
                "csv",
                "model",
                "samples",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
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
            "balanced_accuracy",
            ascending=False
        )

        .groupby(
            [
                "pair",
                "csv"
            ],
            as_index=False
        )

        .first()
    )


    print("\n")
    print("=" * 130)

    print(
        "BEST MODEL PER CSV"
    )

    print("=" * 130)


    print(

        best_per_csv[
            [
                "pair",
                "csv",
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ]
        ].to_string(
            index=False
        )
    )


    best_per_csv.to_csv(
        "unseen_best_per_csv_2026.csv",
        index=False
    )


    # ========================================================
    # BEST MODEL PER PAIR
    # ========================================================

    best_per_pair = (

        results_df

        .sort_values(
            "balanced_accuracy",
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

    print(
        "BEST MODEL PER PAIR"
    )

    print("=" * 130)


    print(

        best_per_pair[
            [
                "pair",
                "csv",
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ]
        ].to_string(
            index=False
        )
    )


    best_per_pair.to_csv(
        "unseen_best_per_pair_2026.csv",
        index=False
    )


else:

    print(
        "\nNo results generated."
    )