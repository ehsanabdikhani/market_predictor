import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

RESULT_ROOT = "train_result_fib_LTF"
MODEL_ROOT = "trained_models"
OUTPUT_ROOT = "feature_analysis"

TARGET = "Movement_Signal"
TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_SAMPLES = 100
N_REPEATS = 10
TOP_N = 30

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def get_original_feature_names(X):
    """Return feature names exactly as they appear in the CSV."""
    return list(X.columns)


def get_transformed_feature_names(preprocessor):
    """
    Get names after ColumnTransformer preprocessing.
    This is useful for native model coefficients/importances.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def aggregate_transformed_importance(values, transformed_names, original_names):
    """
    Aggregate one-hot encoded categorical columns back to the
    original feature name.
    """
    if len(values) != len(transformed_names):
        return pd.DataFrame()

    rows = []

    for name, value in zip(transformed_names, values):
        clean = name

        # Remove sklearn transformer prefix.
        if "__" in clean:
            clean = clean.split("__", 1)[1]

        # For one-hot columns, identify the original column.
        matched = None

        # Exact numeric feature match first.
        for original in original_names:
            if clean == original:
                matched = original
                break

        # Then match categorical one-hot names.
        if matched is None:
            candidates = [
                x for x in original_names
                if clean.startswith(str(x) + "_")
            ]
            if candidates:
                matched = max(candidates, key=len)

        if matched is None:
            matched = clean

        rows.append((matched, float(value)))

    temp = pd.DataFrame(rows, columns=["feature", "importance"])
    return (
        temp.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )


def load_training_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(axis=1, how="all")

    if TARGET not in df.columns:
        return None, None

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    if len(df) < MIN_SAMPLES:
        return None, None

    X = df.drop(columns=[TARGET])

    if "Date" in X.columns:
        X = X.drop(columns=["Date"])

    empty_columns = [
        c for c in X.columns
        if X[c].isna().all()
    ]

    if empty_columns:
        X = X.drop(columns=empty_columns)

    y = df[TARGET]

    return X, y


def native_importance(pipeline, X):
    """
    Extract native importance/coefficients where the underlying
    estimator supports them.
    """
    if "preprocessor" not in pipeline.named_steps:
        return pd.DataFrame()

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    transformed_names = get_transformed_feature_names(preprocessor)
    original_names = get_original_feature_names(X)

    if not transformed_names:
        return pd.DataFrame()

    values = None
    importance_type = None
    direction_values = None

    # LogisticRegression
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)

        if coef.ndim == 2:
            if coef.shape[0] == 1:
                values = np.abs(coef[0])
                direction_values = coef[0]
            else:
                # For multiclass, mean absolute coefficient magnitude.
                values = np.mean(np.abs(coef), axis=0)
                direction_values = np.mean(coef, axis=0)

        importance_type = "coefficient"

    # Tree-based models
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        direction_values = None
        importance_type = "tree_importance"

    if values is None:
        return pd.DataFrame()

    agg = aggregate_transformed_importance(
        values,
        transformed_names,
        original_names
    )

    if agg.empty:
        return agg

    # Add direction for LogisticRegression.
    if direction_values is not None:
        direction_df = aggregate_transformed_importance(
            direction_values,
            transformed_names,
            original_names
        )
        direction_df = direction_df.rename(
            columns={"importance": "signed_effect"}
        )
        agg = agg.merge(direction_df, on="feature", how="left")

        agg["direction"] = np.where(
            agg["signed_effect"] > 0,
            "positive",
            np.where(
                agg["signed_effect"] < 0,
                "negative",
                "neutral"
            )
        )
    else:
        agg["signed_effect"] = np.nan
        agg["direction"] = "nonlinear"

    agg["importance_type"] = importance_type

    total = agg["importance"].sum()
    if total > 0:
        agg["importance_pct"] = (
            agg["importance"] / total * 100
        )
    else:
        agg["importance_pct"] = 0.0

    return agg


def permutation_importance_original_features(
    pipeline,
    X_test,
    y_test
):
    """
    Permutation importance directly on the ORIGINAL input features.

    This is especially useful because the pipeline handles:
      - imputation
      - scaling
      - one-hot encoding
      - model prediction

    Therefore importance is reported using the user's original
    feature names rather than transformed columns.
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    out = pd.DataFrame({
        "feature": X_test.columns,
        "permutation_mean": result.importances_mean,
        "permutation_std": result.importances_std
    })

    # Positive means removing/shuffling the feature hurts
    # balanced accuracy. Negative can happen when a feature
    # is noisy/redundant or permutation variance is high.
    out["importance"] = out["permutation_mean"].clip(lower=0)

    total = out["importance"].sum()
    if total > 0:
        out["importance_pct"] = (
            out["importance"] / total * 100
        )
    else:
        out["importance_pct"] = 0.0

    return out.sort_values(
        "permutation_mean",
        ascending=False
    )


def try_shap_analysis(pipeline, X_test):
    """
    Optional SHAP analysis.

    SHAP is intentionally optional. If it is not installed or a
    model is unsupported, this function returns an empty result.
    """
    try:
        import shap
    except ImportError:
        return pd.DataFrame()

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(X_test)

        # Convert sparse matrices when needed.
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        names = get_transformed_feature_names(preprocessor)

        if not names:
            return pd.DataFrame()

        # Tree models
        tree_models = (
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier"
        )

        if model.__class__.__name__ in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)

        # Linear model
        elif model.__class__.__name__ == "LogisticRegression":
            background = X_transformed
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_transformed)

        else:
            # Kernel SHAP for SVM/KNN can be extremely slow on
            # large FX datasets, so don't run it automatically.
            return pd.DataFrame()

        # Handle binary/multiclass SHAP output.
        if isinstance(shap_values, list):
            arr = np.stack(
                [np.asarray(x) for x in shap_values],
                axis=0
            )
            mean_abs = np.mean(np.abs(arr), axis=(0, 1))
            signed = np.mean(arr, axis=(0, 1))
        else:
            arr = np.asarray(shap_values)

            if arr.ndim == 2:
                mean_abs = np.mean(np.abs(arr), axis=0)
                signed = np.mean(arr, axis=0)
            elif arr.ndim == 3:
                mean_abs = np.mean(np.abs(arr), axis=(0, 2))
                signed = np.mean(arr, axis=(0, 2))
            else:
                return pd.DataFrame()

        out = aggregate_transformed_importance(
            mean_abs,
            names,
            list(X_test.columns)
        )

        signed_df = aggregate_transformed_importance(
            signed,
            names,
            list(X_test.columns)
        )

        signed_df = signed_df.rename(
            columns={"importance": "shap_signed_effect"}
        )

        out = out.merge(
            signed_df[["feature", "shap_signed_effect"]],
            on="feature",
            how="left"
        )

        out["shap_direction"] = np.where(
            out["shap_signed_effect"] > 0,
            "positive",
            np.where(
                out["shap_signed_effect"] < 0,
                "negative",
                "neutral"
            )
        )

        total = out["importance"].sum()
        if total > 0:
            out["importance_pct"] = (
                out["importance"] / total * 100
            )

        return out.sort_values(
            "importance",
            ascending=False
        )

    except Exception as e:
        print(
            "SHAP skipped:",
            type(e).__name__,
            str(e)
        )
        return pd.DataFrame()


# ============================================================
# ANALYZE ONE MODEL
# ============================================================

def analyze_model(model_path, csv_path):
    model_name = os.path.basename(model_path)
    pair = os.path.basename(os.path.dirname(model_path))

    print("\n" + "=" * 100)
    print("MODEL:", model_name)
    print("PAIR :", pair)
    print("=" * 100)

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        print("Could not load:", e)
        return None

    X, y = load_training_data(csv_path)

    if X is None:
        print("Could not load compatible training data:", csv_path)
        return None

    # EXACT same split settings as the training script.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    pred = pipeline.predict(X_test)
    baseline = balanced_accuracy_score(y_test, pred)

    print("Baseline balanced accuracy:", round(baseline, 6))

    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    perm = permutation_importance_original_features(
        pipeline,
        X_test,
        y_test
    )

    perm["pair"] = pair
    perm["model"] = model_name
    perm["baseline_balanced_accuracy"] = baseline

    # --------------------------------------------------------
    # NATIVE IMPORTANCE
    # --------------------------------------------------------

    native = native_importance(
        pipeline,
        X
    )

    if not native.empty:
        native["pair"] = pair
        native["model"] = model_name

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_df = try_shap_analysis(
        pipeline,
        X_test
    )

    if not shap_df.empty:
        shap_df["pair"] = pair
        shap_df["model"] = model_name

    # --------------------------------------------------------
    # COMBINED TABLE
    # --------------------------------------------------------

    combined = perm[
        [
            "pair",
            "model",
            "feature",
            "permutation_mean",
            "permutation_std",
            "importance_pct"
        ]
    ].copy()

    if not native.empty:
        native_small = native[
            [
                "feature",
                "importance",
                "importance_pct",
                "signed_effect",
                "direction",
                "importance_type"
            ]
        ].copy()

        native_small = native_small.rename(
            columns={
                "importance": "native_importance",
                "importance_pct": "native_importance_pct"
            }
        )

        combined = combined.merge(
            native_small,
            on="feature",
            how="left"
        )

    if not shap_df.empty:
        shap_small = shap_df[
            [
                "feature",
                "importance",
                "importance_pct",
                "shap_signed_effect",
                "shap_direction"
            ]
        ].copy()

        shap_small = shap_small.rename(
            columns={
                "importance": "shap_importance",
                "importance_pct": "shap_importance_pct"
            }
        )

        combined = combined.merge(
            shap_small,
            on="feature",
            how="left"
        )

    combined["baseline_balanced_accuracy"] = baseline

    # A rank based on permutation importance is the main ranking.
    combined["permutation_rank"] = (
        combined["permutation_mean"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    safe_name = (
        model_name
        .replace(".joblib", "")
        .replace("/", "_")
        .replace("\\", "_")
    )

    pair_dir = os.path.join(
        OUTPUT_ROOT,
        pair
    )

    os.makedirs(pair_dir, exist_ok=True)

    perm_path = os.path.join(
        pair_dir,
        safe_name + "_permutation.csv"
    )

    native_path = os.path.join(
        pair_dir,
        safe_name + "_native.csv"
    )

    combined_path = os.path.join(
        pair_dir,
        safe_name + "_combined.csv"
    )

    perm.to_csv(
        perm_path,
        index=False
    )

    if not native.empty:
        native.to_csv(
            native_path,
            index=False
        )

    combined.to_csv(
        combined_path,
        index=False
    )

    # --------------------------------------------------------
    # PRINT TOP FEATURES
    # --------------------------------------------------------

    print("\nTOP FEATURES BY PERMUTATION IMPORTANCE")
    print(
        combined[
            [
                "feature",
                "permutation_mean",
                "permutation_std",
                "importance_pct"
            ]
        ]
        .head(TOP_N)
        .to_string(index=False)
    )

    if not native.empty:
        print("\nTOP FEATURES BY NATIVE MODEL IMPORTANCE")
        print(
            native[
                [
                    "feature",
                    "importance_pct",
                    "direction"
                ]
            ]
            .head(TOP_N)
            .to_string(index=False)
        )

    if not shap_df.empty:
        print("\nTOP FEATURES BY SHAP")
        print(
            shap_df[
                [
                    "feature",
                    "importance_pct",
                    "shap_signed_effect",
                    "shap_direction"
                ]
            ]
            .head(TOP_N)
            .to_string(index=False)
        )

    return {
        "pair": pair,
        "model": model_name,
        "baseline_balanced_accuracy": baseline,
        "permutation": perm,
        "native": native,
        "shap": shap_df,
        "combined": combined
    }


# ============================================================
# FIND MODEL / CSV PAIRS
# ============================================================

def find_csv_for_pair(pair, model_filename):
    """
    Your training code loads result/<pair>/*.csv and saves
    models into trained_models/<pair>/.
    """
    candidates = glob.glob(
        os.path.join(
            RESULT_ROOT,
            pair,
            "*.csv"
        )
    )

    if not candidates:
        return None

    # If only one CSV exists, use it.
    if len(candidates) == 1:
        return candidates[0]

    # Try to infer original CSV name from model filename.
    base = model_filename.replace(".joblib", "")

    for csv in candidates:
        stem = os.path.splitext(
            os.path.basename(csv)
        )[0]

        if stem in base:
            return csv

    return candidates[0]


# ============================================================
# RUN ALL MODELS
# ============================================================

all_permutation = []
all_native = []
all_shap = []
all_combined = []

model_files = glob.glob(
    os.path.join(
        MODEL_ROOT,
        "*",
        "*.joblib"
    )
)

print("Found", len(model_files), "trained models.")

if len(model_files) == 0:
    print(
        "\nNo .joblib models found under:",
        MODEL_ROOT
    )
    print(
        "Expected structure: trained_models/<pair>/*.joblib"
    )
else:

    for model_path in model_files:

        pair = os.path.basename(
            os.path.dirname(model_path)
        )

        model_filename = os.path.basename(
            model_path
        )

        csv_path = find_csv_for_pair(
            pair,
            model_filename
        )

        if csv_path is None:
            print(
                "\nNO CSV FOUND FOR:",
                model_path
            )
            continue

        result = analyze_model(
            model_path,
            csv_path
        )

        if result is None:
            continue

        all_permutation.append(
            result["permutation"]
        )

        if not result["native"].empty:
            all_native.append(
                result["native"]
            )

        if not result["shap"].empty:
            all_shap.append(
                result["shap"]
            )

        all_combined.append(
            result["combined"]
        )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

if all_permutation:

    permutation_all = pd.concat(
        all_permutation,
        ignore_index=True
    )

    permutation_all.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "ALL_models_permutation.csv"
        ),
        index=False
    )

    combined_all = pd.concat(
        all_combined,
        ignore_index=True
    )

    combined_all.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "ALL_models_combined.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # FEATURE CONSISTENCY ACROSS MODELS
    # --------------------------------------------------------

    consistency = (
        permutation_all
        .groupby("feature")
        .agg(
            models=("model", "count"),
            mean_importance=("permutation_mean", "mean"),
            median_importance=("permutation_mean", "median"),
            mean_std=("permutation_std", "mean")
        )
        .reset_index()
        .sort_values(
            "mean_importance",
            ascending=False
        )
    )

    consistency["importance_pct"] = (
        consistency["mean_importance"]
        / consistency["mean_importance"].clip(lower=0).sum()
        * 100
    )

    consistency.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "FEATURE_CONSISTENCY_ACROSS_MODELS.csv"
        ),
        index=False
    )

    print("\n" + "=" * 100)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 100)

    print(
        consistency.head(TOP_N).to_string(
            index=False
        )
    )

if all_native:

    native_all = pd.concat(
        all_native,
        ignore_index=True
    )

    native_all.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "ALL_models_native_importance.csv"
        ),
        index=False
    )

if all_shap:

    shap_all = pd.concat(
        all_shap,
        ignore_index=True
    )

    shap_all.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "ALL_models_SHAP.csv"
        ),
        index=False
    )

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print("\nOutput directory:")
print(os.path.abspath(OUTPUT_ROOT))

print("\nMain files:")
print("  ALL_models_permutation.csv")
print("  ALL_models_combined.csv")
print("  ALL_models_native_importance.csv  (when available)")
print("  ALL_models_SHAP.csv               (when SHAP is installed)")
print("  FEATURE_CONSISTENCY_ACROSS_MODELS.csv")

print("\nInterpretation:")
print("- permutation_mean > 0: shuffling the feature hurts balanced accuracy")
print("- larger permutation_mean: more useful predictive feature")
print("- negative permutation_mean: potentially noisy/redundant feature")
print("- LogisticRegression signed_effect gives coefficient direction")
print("- SHAP signed effect gives average prediction direction")
print("- tree native importance measures split-based importance")
print("- permutation importance is the main cross-model comparison")
