import os
import glob
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

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

MODEL_ROOT = "trained_models"

TARGET = "Movement_Signal"

TEST_SIZE = 0.10

CV_SPLITS = 3

N_ITER = 30

RANDOM_STATE = 42

MIN_SAMPLES = 100


# ============================================================
# MODELS + HYPERPARAMETERS
# ============================================================

models = {

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

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


    # ========================================================
    # RANDOM FOREST
    # ========================================================

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
                20,
                30
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


    # ========================================================
    # EXTRA TREES
    # ========================================================

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
                20,
                30
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


    # ========================================================
    # GRADIENT BOOSTING
    # ========================================================

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


    # ========================================================
    # HISTOGRAM GRADIENT BOOSTING
    # ========================================================

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


    # ========================================================
    # SVM
    # ========================================================

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


    # ========================================================
    # KNN
    # ========================================================

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

def create_pipeline(model, X):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_features:

        transformers.append(
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

                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
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

                categorical_features
            )
        )

    return Pipeline([
        (
            "preprocessor",

            ColumnTransformer(
                transformers=transformers
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

    prediction = model.predict(
        X_test
    )

    result = {}


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    result["accuracy"] = accuracy_score(
        y_test,
        prediction
    )


    # --------------------------------------------------------
    # BALANCED ACCURACY
    # --------------------------------------------------------

    result["balanced_accuracy"] = (
        balanced_accuracy_score(
            y_test,
            prediction
        )
    )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    result["precision"] = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    result["recall"] = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    result["f1"] = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    try:

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        if len(model.classes_) == 2:

            result["roc_auc"] = (
                roc_auc_score(
                    y_test,
                    probabilities[:, 1]
                )
            )

        else:

            result["roc_auc"] = (
                roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted"
                )
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


    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    pair = os.path.basename(
        os.path.dirname(filepath)
    )

    filename = os.path.basename(
        filepath
    )


    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        filepath
    )


    df = df.dropna(
        axis=1,
        how="all"
    )


    if TARGET not in df.columns:

        print(
            "TARGET NOT FOUND:",
            TARGET
        )

        return []


    # --------------------------------------------------------
    # REMOVE MISSING TARGET
    # --------------------------------------------------------

    df = df.dropna(
        subset=[TARGET]
    ).reset_index(
        drop=True
    )


    if len(df) < MIN_SAMPLES:

        print(
            "SKIPPED - only",
            len(df),
            "samples"
        )

        return []


    # ========================================================
    # FEATURES
    # ========================================================

    # --------------------------------------------------------
    # SORT DATA BY DATE (OLDER -> NEWER)
    # --------------------------------------------------------

    if "Date" not in df.columns:

        print(
            "DATE COLUMN NOT FOUND"
        )

        return []


    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY COLUMNS
    # --------------------------------------------------------

    feature_columns = [

        c for c in df.columns

        if c not in [TARGET, "Date"]

        and not df[c].isna().all()
    ]


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    test_count = int(
        len(df) * TEST_SIZE
    )


    train_df = df.iloc[
        :-test_count
    ]

    test_df = df.iloc[
        -test_count:
    ]


    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[TARGET]


    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[TARGET]


    print(
        "Samples:",
        len(df)
    )

    print(
        "Train:",
        len(X_train),
        "| Test:",
        len(X_test)
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df[TARGET].value_counts()
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


        # ----------------------------------------------------
        # MODEL SAVE PATH
        # ----------------------------------------------------

        model_dir = os.path.join(
            MODEL_ROOT,
            pair
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )


        model_filename = (
            filename.replace(
                ".csv",
                ""
            )
            + "_"
            + model_name
            + ".joblib"
        )


        model_path = os.path.join(
            model_dir,
            model_filename
        )


        # ====================================================
        # CHECK IF ALREADY TRAINED
        # ====================================================

        if os.path.exists(
            model_path
        ):

            print(
                "MODEL ALREADY EXISTS:"
            )

            print(
                model_path
            )

            print(
                "Skipping training."
            )

            continue


        # ----------------------------------------------------
        # CREATE PIPELINE
        # ----------------------------------------------------

        pipeline = create_pipeline(
            base_model,
            X_train
        )


        # ====================================================
        # RANDOMIZED SEARCH
        # ====================================================

        search = RandomizedSearchCV(

            estimator=pipeline,

            param_distributions=params,

            n_iter=N_ITER,

            scoring="balanced_accuracy",

            cv=cv,

            n_jobs=-1,

            verbose=0,

            random_state=RANDOM_STATE,

            refit=True
        )


        # ====================================================
        # TRAIN
        # ====================================================

        try:

            search.fit(
                X_train,
                y_train
            )

        except Exception as e:

            print(
                "ERROR:",
                model_name
            )

            print(
                e
            )

            continue


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model = (
            search.best_estimator_
        )


        # ====================================================
        # SAVE MODEL
        # ====================================================

        joblib.dump(

            best_model,

            model_path
        )


        print(
            "MODEL SAVED:"
        )

        print(
            model_path
        )


        # ====================================================
        # TEST
        # ====================================================

        metrics, prediction = (
            calculate_metrics(

                best_model,

                X_test,

                y_test
            )
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(

            y_test,

            prediction,

            labels=best_model.classes_
        )


        # ====================================================
        # RESULT
        # ====================================================

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
                metrics[
                    "balanced_accuracy"
                ],

            "test_precision":
                metrics[
                    "precision"
                ],

            "test_recall":
                metrics[
                    "recall"
                ],

            "test_f1":
                metrics[
                    "f1"
                ],

            "test_roc_auc":
                metrics[
                    "roc_auc"
                ],

            "best_params":
                str(
                    search.best_params_
                ),

            "confusion_matrix":
                str(cm),

            "model_path":
                model_path
        }


        all_results.append(
            result
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "\nCV balanced accuracy:",
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
                metrics[
                    "balanced_accuracy"
                ],
                4
            )
        )

        print(
            "TEST precision:",
            round(
                metrics[
                    "precision"
                ],
                4
            )
        )

        print(
            "TEST recall:",
            round(
                metrics[
                    "recall"
                ],
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

        if np.isnan(
            metrics["roc_auc"]
        ):

            print(
                "TEST ROC-AUC: N/A"
            )

        else:

            print(
                "TEST ROC-AUC:",
                round(
                    metrics["roc_auc"],
                    4
                )
            )


        print(
            "\nBEST PARAMETERS:"
        )

        print(
            search.best_params_
        )


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
# RUN ALL FILES
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

        "ml_results.csv",

        index=False
    )


    # ========================================================
    # PRINT ALL
    # ========================================================

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
                "model_path"
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
                "test_roc_auc",
                "model_path"
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
        "\nNo new models were trained."
    )