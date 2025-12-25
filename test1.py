"""
Pipeline dự báo bỏ học cho hệ thống LMS. 
Được tách hàm để dùng lại cho cron job hằng ngày (không cần notebook).
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import PowerTransformer


# -----------------
# Config
# -----------------
DATA_DIR = "datasets/"
MODULE = "BBB"
PRESENTATIONS = ["2013B", "2013J"]
TIME_CHECKPOINTS = [14, 30, 60, 90, 120, 150, 180, 210]
WINDOW_DAYS = 14
HALF_WINDOW = 7
VAR_THRESH = 0.0  # an toàn cho sản xuất
MODEL_PATH = "dropout_model_time_aware.pkl"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def compute_inactivity_streak(days_list: List[int], start_day: int, end_day: int) -> int:
    """Đếm số ngày cuối cùng không hoạt động trong đoạn [start_day, end_day]."""
    if not days_list:
        return end_day - start_day + 1
    active_set = set(days_list)
    streak = 0
    d = end_day
    while d >= start_day and d not in active_set:
        streak += 1
        d -= 1
    return streak


def load_raw_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        "student_info": pd.read_csv(data_dir / "studentInfo.csv"),
        "student_reg": pd.read_csv(data_dir / "studentRegistration.csv"),
        "student_vle": pd.read_csv(data_dir / "studentVle.csv"),
        "student_ass": pd.read_csv(data_dir / "studentAssessment.csv"),
        "assessments": pd.read_csv(data_dir / "assessments.csv"),
    }


def filter_students(raw: Dict[str, pd.DataFrame], module: str, presentations: List[str]) -> pd.DataFrame:
    reg_filtered = raw["student_reg"][
        (raw["student_reg"]["code_module"] == module)
        & (raw["student_reg"]["code_presentation"].isin(presentations))
        & (raw["student_reg"]["date_registration"] <= 0)
    ]
    valid_ids = reg_filtered["id_student"].unique()

    students = raw["student_info"][
        (raw["student_info"]["code_module"] == module)
        & (raw["student_info"]["code_presentation"].isin(presentations))
        & (raw["student_info"]["id_student"].isin(valid_ids))
    ].copy()
    students["dropout"] = (students["final_result"] == "Withdrawn").astype(int)
    return students


def build_features(
    students: pd.DataFrame,
    raw: Dict[str, pd.DataFrame],
    module: str,
    presentations: List[str],
    time_checkpoints: List[int],
    window_days: int = WINDOW_DAYS,
    half_window: int = HALF_WINDOW,
) -> Tuple[pd.DataFrame, List[str]]:
    assessments = raw["assessments"]
    target_assessment_ids = assessments.loc[
        (assessments["code_module"] == module)
        & (assessments["code_presentation"].isin(presentations)),
        "id_assessment",
    ].unique()

    augmented = []
    for cutoff in time_checkpoints:
        w_start = max(0, cutoff - (window_days - 1))
        w_end = cutoff

        vle_cum = raw["student_vle"][
            (raw["student_vle"]["code_module"] == module)
            & (raw["student_vle"]["code_presentation"].isin(presentations))
            & (raw["student_vle"]["id_student"].isin(students["id_student"]))
            & (raw["student_vle"]["date"] <= cutoff)
        ].copy()

        vle_win = vle_cum[vle_cum["date"] >= w_start].copy()

        cum_agg = (
            vle_cum.groupby("id_student")
            .agg(
                total_clicks=("sum_click", "sum"),
                active_days_total=("date", "nunique"),
                last_active=("date", "max"),
            )
            .reset_index()
        )
        cum_agg["days_elapsed_program"] = cutoff
        cum_agg["clicks_per_day_total"] = cum_agg["total_clicks"] / max(cutoff, 1)
        cum_agg["active_ratio_total"] = cum_agg["active_days_total"] / max(cutoff, 1)
        cum_agg["days_since_last_active"] = cutoff - cum_agg["last_active"]
        cum_agg["avg_clicks_per_active_day_total"] = (
            cum_agg["total_clicks"] / cum_agg["active_days_total"].replace(0, np.nan)
        ).fillna(0)

        win_agg = (
            vle_win.groupby("id_student")
            .agg(clicks_last_14_days=("sum_click", "sum"), active_days_14=("date", "nunique"))
            .reset_index()
        )
        win_agg["clicks_per_day_14"] = win_agg["clicks_last_14_days"] / window_days
        win_agg["active_ratio_14"] = win_agg["active_days_14"] / window_days

        first_end = min(w_end, w_start + (half_window - 1))
        second_start = min(w_end, first_end + 1)

        clicks_0_7 = (
            vle_win[(vle_win["date"] >= w_start) & (vle_win["date"] <= first_end)]
            .groupby("id_student")["sum_click"]
            .sum()
            .reset_index(name="clicks_0_7")
        )
        clicks_8_14 = (
            vle_win[(vle_win["date"] >= second_start) & (vle_win["date"] <= w_end)]
            .groupby("id_student")["sum_click"]
            .sum()
            .reset_index(name="clicks_8_14")
        )

        clicks_last_7 = (
            vle_cum[vle_cum["date"] > (cutoff - 7)]
            .groupby("id_student")["sum_click"]
            .sum()
            .reset_index(name="clicks_last_7_days")
        )

        days_list = (
            vle_win.groupby("id_student")["date"]
            .apply(lambda x: sorted(x.unique()))
            .reset_index()
            .rename(columns={"date": "active_days_list"})
        )
        days_list["inactivity_streak_14"] = days_list["active_days_list"].apply(
            lambda lst: compute_inactivity_streak(lst, w_start, w_end)
        )
        streak = days_list[["id_student", "inactivity_streak_14"]]

        ass_cum = raw["student_ass"][
            (raw["student_ass"]["id_assessment"].isin(target_assessment_ids))
            & (raw["student_ass"]["id_student"].isin(students["id_student"]))
            & (raw["student_ass"]["date_submitted"].notna())
            & (raw["student_ass"]["date_submitted"] <= cutoff)
        ].copy()

        ass_agg = (
            ass_cum.groupby("id_student")
            .agg(
                num_assessments=("id_assessment", "nunique"),
                avg_score=("score", "mean"),
                pass_count=("score", lambda x: (x >= 40).sum()),
                last_score=("score", "last"),
            )
            .reset_index()
        )

        ass_win = ass_cum[ass_cum["date_submitted"] >= w_start]
        ass_agg_14 = (
            ass_win.groupby("id_student")
            .agg(num_assessments_14=("id_assessment", "nunique"), avg_score_14=("score", "mean"))
            .reset_index()
        )

        base = students[["id_student", "dropout"]].copy()
        merged = base.merge(cum_agg, on="id_student", how="left")
        merged = merged.merge(win_agg, on="id_student", how="left")
        merged = merged.merge(clicks_0_7, on="id_student", how="left")
        merged = merged.merge(clicks_8_14, on="id_student", how="left")
        merged = merged.merge(clicks_last_7, on="id_student", how="left")
        merged = merged.merge(streak, on="id_student", how="left")
        merged = merged.merge(ass_agg, on="id_student", how="left")
        merged = merged.merge(ass_agg_14, on="id_student", how="left")

        fill0 = [
            "total_clicks",
            "active_days_total",
            "last_active",
            "clicks_last_14_days",
            "active_days_14",
            "clicks_0_7",
            "clicks_8_14",
            "clicks_last_7_days",
            "inactivity_streak_14",
            "num_assessments",
            "avg_score",
            "pass_count",
            "last_score",
            "num_assessments_14",
            "avg_score_14",
        ]
        for col in fill0:
            merged[col] = merged[col].fillna(0)

        merged["trend_click_14"] = merged["clicks_8_14"] - merged["clicks_0_7"]
        merged["ratio_click_14"] = (merged["clicks_8_14"] + 1) / (merged["clicks_0_7"] + 1)
        merged["days_elapsed_program"] = cutoff

        augmented.append(merged)

    final_df = pd.concat(augmented, ignore_index=True)

    feature_cols = [
        "days_elapsed_program",
        "clicks_per_day_total",
        "active_ratio_total",
        "avg_clicks_per_active_day_total",
        "days_since_last_active",
        "clicks_last_14_days",
        "active_days_14",
        "clicks_per_day_14",
        "active_ratio_14",
        "clicks_last_7_days",
        "clicks_0_7",
        "clicks_8_14",
        "trend_click_14",
        "ratio_click_14",
        "inactivity_streak_14",
        "num_assessments",
        "avg_score",
        "pass_count",
        "last_score",
        "num_assessments_14",
        "avg_score_14",
    ]

    return final_df, feature_cols


def make_eval_pipe(model):
    return ImbPipeline(
        [
            ("variance_threshold", VarianceThreshold(VAR_THRESH)),
            ("smote", SMOTE()),
            ("power_transformer", PowerTransformer()),
            ("classifier", model),
        ]
    )


def make_prod_pipe(model):
    return SkPipeline(
        [
            ("variance_threshold", VarianceThreshold(VAR_THRESH)),
            ("power_transformer", PowerTransformer()),
            ("classifier", model),
        ]
    )


MODELS = {
    "GradientBoostingClassifier": GradientBoostingClassifier(
        learning_rate=0.03,
        loss="exponential",
        max_depth=3,
        min_samples_leaf=40,
        min_samples_split=20,
        n_estimators=10,
        n_iter_no_change=10,
        random_state=42,
    ),
    "RandomForestClassifier": RandomForestClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_leaf=10,
        min_samples_split=50,
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
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
        random_state=42,
    ),
    "LogisticRegression": LogisticRegression(
        penalty="l1",
        solver="saga",
        tol=1e-4,
        max_iter=2000,
        random_state=42,
    ),
}


def evaluate_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=5)
    summary_rows = []

    for name, model in MODELS.items():
        rows = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            pipe = make_eval_pipe(model)
            pipe.fit(X_tr, y_tr)

            y_pred = pipe.predict(X_te)
            y_proba = pipe.predict_proba(X_te)[:, 1]

            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "accuracy": accuracy_score(y_te, y_pred),
                    "f1": f1_score(y_te, y_pred),
                    "sensitivity": recall_score(y_te, y_pred),
                    "specificity": recall_score(y_te, y_pred, pos_label=0),
                    "auc": roc_auc_score(y_te, y_proba),
                }
            )

        df = pd.DataFrame(rows)
        summary_rows.append(
            {
                "model": name,
                "mean_accuracy": df["accuracy"].mean(),
                "mean_f1": df["f1"].mean(),
                "mean_sensitivity": df["sensitivity"].mean(),
                "mean_specificity": df["specificity"].mean(),
                "mean_auc": df["auc"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_f1", ascending=False)
    logging.info("Hiệu năng trung bình (sort theo F1):\n%s", summary_df)
    return summary_df


def train_and_save(final_df: pd.DataFrame, feature_cols: List[str], model_path: str) -> str:
    X = final_df[feature_cols].fillna(0)
    y = final_df["dropout"].astype(int)
    groups = final_df["id_student"]

    summary_df = evaluate_models(X, y, groups)
    best_model_name = summary_df.iloc[0]["model"]
    logging.info("Chọn model tốt nhất: %s", best_model_name)

    final_model = MODELS[best_model_name]
    pipeline = make_prod_pipe(final_model)
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    logging.info("Đã lưu model vào %s", model_path)
    return best_model_name


def predict_dropout(
    model_path: str,
    students: pd.DataFrame,
    raw: Dict[str, pd.DataFrame],
    cutoff_day: int,
    module: str,
    presentations: List[str],
) -> pd.DataFrame:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    pipeline = joblib.load(model_path)
    final_df, feature_cols = build_features(
        students=students,
        raw=raw,
        module=module,
        presentations=presentations,
        time_checkpoints=[cutoff_day],
    )
    X_today = final_df[feature_cols].fillna(0)
    proba = pipeline.predict_proba(X_today)[:, 1]
    out = final_df[["id_student", "dropout"]].copy()
    out["dropout_proba"] = proba
    out["days_elapsed_program"] = cutoff_day
    return out.sort_values("dropout_proba", ascending=False)


def send_email_stub(student_row: pd.Series) -> None:
    """Placeholder: thay bằng SMTP/Mail service thực tế."""
    logging.info(
        "[EMAIL] Gửi cảnh báo tới học viên %s, xác suất bỏ học=%.2f",
        student_row["id_student"],
        student_row["dropout_proba"],
    )


def daily_job(threshold: float = 0.5, cutoff_day: int = 60) -> None:
    raw = load_raw_data(DATA_DIR)
    students = filter_students(raw, MODULE, PRESENTATIONS)
    alerts = predict_dropout(MODEL_PATH, students, raw, cutoff_day, MODULE, PRESENTATIONS)
    need_alert = alerts[alerts["dropout_proba"] >= threshold]
    for _, row in need_alert.iterrows():
        send_email_stub(row)
    logging.info("Đã xử lý %d học viên, gửi %d cảnh báo", len(alerts), len(need_alert))


def train_pipeline() -> None:
    logging.info("Bắt đầu load dữ liệu và train...")
    raw = load_raw_data(DATA_DIR)
    students = filter_students(raw, MODULE, PRESENTATIONS)
    logging.info("Số học viên hợp lệ: %d", students["id_student"].nunique())
    final_df, feature_cols = build_features(students, raw, MODULE, PRESENTATIONS, TIME_CHECKPOINTS)
    logging.info("Dataset augmented: %s", final_df.shape)
    best_model = train_and_save(final_df, feature_cols, MODEL_PATH)
    logging.info("Hoàn tất train, best model: %s", best_model)


if __name__ == "__main__":
    # Chạy train một lần để tạo model phục vụ cron.
    train_pipeline()
    # Ví dụ chạy cron hằng ngày: daily_job(threshold=0.45, cutoff_day=120)