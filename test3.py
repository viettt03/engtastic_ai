import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score
)

# =========================================================
# 1) LOAD
# =========================================================
DATA_DIR = "datasets/"
MODULE = "BBB"
PRESENTATIONS = ["2013B", "2013J"]
TIME_CHECKPOINTS = [14, 30, 60, 90, 120, 150, 180, 210]

student_info = pd.read_csv(DATA_DIR + "studentInfo.csv")
student_reg  = pd.read_csv(DATA_DIR + "studentRegistration.csv")
student_vle  = pd.read_csv(DATA_DIR + "studentVle.csv")
student_ass  = pd.read_csv(DATA_DIR + "studentAssessment.csv")

print(f"--- {MODULE} {PRESENTATIONS} ---")
print(f"TIME CHECKPOINTS: {TIME_CHECKPOINTS}")

# =========================================================
# 2) FILTER STUDENTS (self-paced simulation)
# =========================================================
reg_filtered = student_reg[
    (student_reg["code_module"] == MODULE) &
    (student_reg["code_presentation"].isin(PRESENTATIONS)) &
    (student_reg["date_registration"] <= 0)
]
valid_ids = reg_filtered["id_student"].unique()

students = student_info[
    (student_info["code_module"] == MODULE) &
    (student_info["code_presentation"].isin(PRESENTATIONS)) &
    (student_info["id_student"].isin(valid_ids))
].copy()

students["dropout"] = (students["final_result"] == "Withdrawn").astype(int)
print("Unique students:", students["id_student"].nunique())

# =========================================================
# 3) AUGMENT DATASET BY CHECKPOINTS
# =========================================================
augmented_data_list = []

for cutoff in TIME_CHECKPOINTS:
    # --- VLE snapshot ---
    vle_snapshot = student_vle[
        (student_vle["code_module"] == MODULE) &
        (student_vle["code_presentation"].isin(PRESENTATIONS)) &
        (student_vle["id_student"].isin(students["id_student"])) &
        (student_vle["date"] <= cutoff)
    ]

    vle_agg = vle_snapshot.groupby("id_student").agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        last_active=("date", "max")
    ).reset_index()

    # feature engineering
    vle_agg["days_elapsed_program"] = cutoff
    vle_agg["clicks_per_day"] = vle_agg["total_clicks"] / cutoff
    vle_agg["active_ratio"] = vle_agg["active_days"] / cutoff
    vle_agg["days_since_last_active"] = cutoff - vle_agg["last_active"]

    recent_clicks = (
        vle_snapshot[vle_snapshot["date"] > (cutoff - 7)]
        .groupby("id_student")["sum_click"]
        .sum()
        .reset_index(name="clicks_last_7_days")
    )
    vle_agg = vle_agg.merge(recent_clicks, on="id_student", how="left").fillna(0)

    # --- Assessment snapshot (FIX LEAKAGE: filter module/presentation) ---
    ass_snapshot = student_ass[
        (student_ass["code_module"] == MODULE) &
        (student_ass["code_presentation"].isin(PRESENTATIONS)) &
        (student_ass["id_student"].isin(students["id_student"])) &
        (student_ass["date_submitted"].notna()) &
        (student_ass["date_submitted"] <= cutoff)
    ]

    ass_agg = ass_snapshot.groupby("id_student").agg(
        num_assessments=("id_assessment", "nunique"),
        avg_score=("score", "mean"),
        pass_count=("score", lambda x: (x >= 40).sum())
    ).reset_index()

    merged = students[["id_student", "dropout"]].merge(vle_agg, on="id_student", how="left")
    merged = merged.merge(ass_agg, on="id_student", how="left")

    # fill missing
    merged["days_elapsed_program"] = cutoff
    merged["total_clicks"] = merged["total_clicks"].fillna(0)
    merged["clicks_per_day"] = merged["clicks_per_day"].fillna(0)
    merged["active_ratio"] = merged["active_ratio"].fillna(0)
    merged["days_since_last_active"] = merged["days_since_last_active"].fillna(cutoff)
    merged["clicks_last_7_days"] = merged["clicks_last_7_days"].fillna(0)

    merged["num_assessments"] = merged["num_assessments"].fillna(0)
    merged["avg_score"] = merged["avg_score"].fillna(0)
    merged["pass_count"] = merged["pass_count"].fillna(0)

    augmented_data_list.append(merged)

final_df = pd.concat(augmented_data_list, ignore_index=True)
print("Augmented shape:", final_df.shape)

# =========================================================
# 4) FEATURES (NUMERIC ONLY - to match notebook style)
# =========================================================
feature_cols_num = [
    "days_elapsed_program",
    "total_clicks", "clicks_per_day", "active_ratio",
    "days_since_last_active", "clicks_last_7_days",
    "num_assessments", "avg_score", "pass_count"
]

X = final_df[feature_cols_num]
y = final_df["dropout"].astype(int)
groups = final_df["id_student"]

# =========================================================
# 5) 4 MODELS + HYPERPARAMS EXACTLY AS YOU SENT
# =========================================================
MODELS = {
    "GradientBoostingClassifier": GradientBoostingClassifier(
        learning_rate=0.03,
        loss="exponential",
        max_depth=3,
        min_samples_leaf=40,
        min_samples_split=20,
        n_estimators=10,
        n_iter_no_change=10,
        random_state=42
    ),
    "RandomForestClassifier": RandomForestClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_leaf=10,
        min_samples_split=50,
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),
    "MLPClassifier": MLPClassifier(
        alpha=0.1,
        early_stopping=True,
        hidden_layer_sizes=(135,),
        learning_rate="constant",
        learning_rate_init=0.3,
        max_iter=1200,
        momentum=0.9,
        solver="sgd",
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        penalty="l1",
        solver="saga",
        tol=1e-4,
        max_iter=2000,
        random_state=42
    )
}

def evaluate_groupkfold(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        pipe = Pipeline([
            ("variance_threshold", VarianceThreshold(3)),
            ("smote", SMOTE()),                 # giống link
            ("power_transformer", PowerTransformer()),
            ("classifier", model),
        ])

        pipe.fit(X_tr, y_tr)

        # predict
        y_pred = pipe.predict(X_te)

        # AUC: cần proba nếu có
        if hasattr(pipe.named_steps["classifier"], "predict_proba"):
            y_proba = pipe.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_proba)
        else:
            # fallback (hiếm)
            auc = np.nan

        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred)
        sens = recall_score(y_te, y_pred)            # sensitivity = recall for class 1
        spec = recall_score(y_te, y_pred, pos_label=0)

        rows.append({
            "fold": fold,
            "accuracy": acc,
            "f1": f1,
            "sensitivity": sens,
            "specificity": spec,
            "auc": auc
        })

    df = pd.DataFrame(rows)
    return df, df.mean(numeric_only=True)

print("\n===== 5-Fold GroupKFold Results =====")
summary = []
for name, model in MODELS.items():
    fold_df, mean_row = evaluate_groupkfold(model, X, y, groups, n_splits=5)
    mean_row = mean_row.to_dict()
    mean_row["model"] = name
    summary.append(mean_row)

summary_df = pd.DataFrame(summary).sort_values("f1", ascending=False)
print(summary_df[["model","accuracy","f1","sensitivity","specificity","auc"]].to_string(index=False))

best_model_name = summary_df.iloc[0]["model"]
print("\nBest model by mean F1:", best_model_name)

# =========================================================
# 6) TRAIN FINAL PIPELINE FOR PRODUCTION (fit on ALL data)
# =========================================================
best_model = MODELS[best_model_name]

final_pipeline = Pipeline([
    ("variance_threshold", VarianceThreshold(3)),
    ("smote", SMOTE()),
    ("power_transformer", PowerTransformer()),
    ("classifier", best_model),
])

final_pipeline.fit(X, y)

# production rule: dùng predict_proba (nếu có) + threshold mặc định 0.5
PROD_THRESHOLD = 0.5

def predict_dropout_risk(feature_row_df: pd.DataFrame) -> dict:
    proba = None
    if hasattr(final_pipeline.named_steps["classifier"], "predict_proba"):
        proba = float(final_pipeline.predict_proba(feature_row_df)[0, 1])
        label = int(proba >= PROD_THRESHOLD)
    else:
        label = int(final_pipeline.predict(feature_row_df)[0])
        proba = None
    return {"risk_prob": proba, "risk_label": label, "threshold": PROD_THRESHOLD}

# Example (Cron simulation)
sample = pd.DataFrame([{
    "days_elapsed_program": 45,
    "total_clicks": 120,
    "clicks_per_day": 120/45,
    "active_ratio": 10/45,
    "days_since_last_active": 5,
    "clicks_last_7_days": 2,
    "num_assessments": 1,
    "avg_score": 80,
    "pass_count": 1
}])

print("\nCron simulation:", predict_dropout_risk(sample))
