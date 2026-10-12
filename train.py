import os
import glob
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

RESULT_ROOT = "result"

TARGET = "Movement_Signal"

TEST_SIZE = 0.20

CV_SPLITS = 5

RANDOM_STATE = 42

MIN_SAMPLES = 100


# ============================================================
# MODELS + HYPERPARAMETERS
# ============================================================

models = {

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    "LogisticRegression": (

        LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
        ),

        {
            "model__C": [
                0.001,
                0.01,
                0.1,
                0.5,
                1,
                2,
                10,
                100
            ],

            "model__solver": [
                "lbfgs",
                "liblinear"
            ]
        }
    ),


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    "RandomForest": (

        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),

        {
            "model__n_estimators": [
                200,
                500,
                1000
            ],

            "model__max_depth": [
                None,
                3,
                5,
                10,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                5,
                10,
                20
            ],

            "model__max_features": [
                "sqrt",
                "log2",
                None
            ]
        }
    ),


    # --------------------------------------------------------
    # EXTRA TREES
    # --------------------------------------------------------

    "ExtraTrees": (

        ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),

        {
            "model__n_estimators": [
                200,
                500,
                1000
            ],

            "model__max_depth": [
                None,
                3,
                5,
                10,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                5,
                10,
                20
            ],

            "model__max_features": [
                "sqrt",
                "log2",
                None
            ]
        }
    ),


    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    "GradientBoosting": (

        GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        {
            "model__n_estimators": [
                50,
                100,
                200,
                400
            ],

            "model__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1
            ],

            "model__max_depth": [
                1,
                2,
                3,
                5
            ],

            "model__min_samples_leaf": [
                2,
                5,
                10,
                20
            ]
        }
    ),


    # --------------------------------------------------------
    # HIST GRADIENT BOOSTING
    # --------------------------------------------------------

    "HistGradientBoosting": (

        HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        {
            "model__max_iter": [
                100,
                200,
                400
            ],

            "model__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1
            ],

            "model__max_leaf_nodes": [
                7,
                15,
                31,
                63
            ],

            "model__min_samples_leaf": [
                5,
                10,
                20,
                30
            ]
        }
    ),


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    "SVM": (

        SVC(
            probability=True,
            class_weight="balanced"
        ),

        {
            "model__C": [
                0.01,
                0.1,
                1,
                10,
                100
            ],

            "model__gamma": [
                "scale",
                "auto",
                0.001,
                0.01,
                0.1
            ],

            "model__kernel": [
                "rbf"
            ]
        }
    ),


    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    "KNN": (

        KNeighborsClassifier(
            n_jobs=-1
        ),

        {
            "model__n_neighbors": [
                3,
                5,
                10,
                20,
                30,
                50
            ],

            "model__weights": [
                "uniform",
                "distance"
            ],

            "model__p": [
                1,
                2
            ]
        }
    )
}


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

def create_pipeline(model):

    return Pipeline([

        (
            "preprocessor",

            ColumnTransformer(

                transformers=[

                    # -----------------------------
                    # NUMERIC
                    # -----------------------------

                    (
                        "numeric",

                        Pipeline([

                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median"
                                )
                            ),

                            (
                                "scaler",
                                StandardScaler()
                            )

                        ]),

                        lambda df:
                        df.select_dtypes(
                            include=np.number
                        ).columns.tolist()
                    ),


                    # -----------------------------
                    # CATEGORICAL
                    # -----------------------------

                    (
                        "categorical",

                        Pipeline([

                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                )
                            ),

                            (
                                "onehot",

                                OneHotEncoder(
                                    handle_unknown="ignore"
                                )
                            )

                        ]),

                        lambda df:
                        df.select_dtypes(
                            exclude=np.number
                        ).columns.tolist()
                    )
                ]
            )
        ),

        (
            "model",
            model
        )
    ])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    model,
    X_test,
    y_test
):

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

        if len(model.classes_) == 2:

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

    except:

        result["roc_auc"] = np.nan


    return result, prediction


# ============================================================
# PROCESS ONE CSV
# ============================================================

def process_file(filepath):

    print("\n")
    print("=" * 100)
    print("FILE:", filepath)
    print("=" * 100)

    df = pd.read_csv(filepath)

    df = df.dropna(
        axis=1,
        how="all"
    )

    if TARGET not in df.columns:

        print(
            "TARGET NOT FOUND"
        )

        return []


    # --------------------------------------------------------
    # Remove missing target
    # --------------------------------------------------------

    df = df.dropna(
        subset=[TARGET]
    ).reset_index(drop=True)


    if len(df) < MIN_SAMPLES:

        print(
            "SKIPPED:",
            len(df),
            "samples"
        )

        return []


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]


    # --------------------------------------------------------
    # DATE IS NOT A FEATURE
    # --------------------------------------------------------

    if "Date" in X.columns:

        X = X.drop(
            columns=["Date"]
        )


    # --------------------------------------------------------
    # Remove completely empty columns
    # --------------------------------------------------------

    empty_columns = [
        c for c in X.columns
        if X[c].isna().all()
    ]

    if empty_columns:

        X = X.drop(
            columns=empty_columns
        )


    # --------------------------------------------------------
    # RANDOM STRATIFIED SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )


    print(
        "Samples:",
        len(df)
    )

    print(
        "Train:",
        len(X_train),
        " Test:",
        len(X_test)
    )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
    )


    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    cv = StratifiedKFold(

        n_splits=CV_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE
    )


    all_results = []


    # ========================================================
    # EVERY MODEL
    # ========================================================

    for model_name, (
        base_model,
        params
    ) in models.items():

        print("\n")
        print(
            "-" * 80
        )

        print(
            "MODEL:",
            model_name
        )

        print(
            "-" * 80
        )


        pipeline = create_pipeline(
            base_model
        )


        search = RandomizedSearchCV(

            estimator=pipeline,

            param_distributions=params,

            n_iter=25,

            cv=3,

            scoring='balanced_accuracy',

            random_state=42,

            refit=True
        )


        try:

            search.fit(
                X_train,
                y_train
            )

        except Exception as e:

            print(
                "ERROR:",
                e
            )

            continue


        best_model = (
            search.best_estimator_
        )


        metrics, prediction = (
            calculate_metrics(
                best_model,
                X_test,
                y_test
            )
        )


        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            prediction,
            labels=best_model.classes_
        )


        # ----------------------------------------------------
        # File information
        # ----------------------------------------------------

        pair = os.path.basename(
            os.path.dirname(filepath)
        )

        filename = os.path.basename(
            filepath
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "pair":
                pair,

            "file":
                filename,

            "model":
                model_name,

            "samples":
                len(df),

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test),

            "cv_balanced_accuracy":
                search.best_score_,

            "test_accuracy":
                metrics["accuracy"],

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
                str(search.best_params_),

            "confusion_matrix":
                str(cm)
        }


        all_results.append(
            result
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            "CV balanced accuracy:",
            round(
                search.best_score_,
                4
            )
        )

        print(
            "TEST accuracy:",
            round(
                metrics["accuracy"],
                4
            )
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

        print(
            "TEST ROC-AUC:",
            round(
                metrics["roc_auc"],
                4
            )
            if not np.isnan(
                metrics["roc_auc"]
            )
            else "N/A"
        )

        print(
            "BEST PARAMS:"
        )

        print(
            search.best_params_
        )


    return all_results


# ============================================================
# FIND ALL CSVs
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
# RUN
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

    # --------------------------------------------------------
    # ALL RESULTS
    # --------------------------------------------------------

    results_df = results_df.sort_values(

        [
            "test_balanced_accuracy",
            "test_f1",
            "test_accuracy"
        ],

        ascending=False
    )


    results_df.to_csv(
        "ml_results.csv",
        index=False
    )


    # --------------------------------------------------------
    # PRINT EVERYTHING
    # --------------------------------------------------------

    print("\n")
    print(
        "=" * 130
    )

    print(
        "ALL MODEL RESULTS"
    )

    print(
        "=" * 130
    )


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
    print(
        "=" * 130
    )

    print(
        "BEST MODEL PER CSV"
    )

    print(
        "=" * 130
    )


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
                "best_params"
            ]
        ].to_string(
            index=False
        )
    )


    best_per_csv.to_csv(
        "ml_best_per_csv.csv",
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
    print(
        "=" * 130
    )

    print(
        "BEST MODEL PER PAIR"
    )

    print(
        "=" * 130
    )


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
                "test_roc_auc"
            ]
        ].to_string(
            index=False
        )
    )


    best_per_pair.to_csv(
        "ml_best_per_pair.csv",
        index=False
    )


else:

    print(
        "No results generated."
    )