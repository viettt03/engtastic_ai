"""
API dự đoán khả năng bỏ học của học viên
Backend chỉ cần truyền dữ liệu thô, API sẽ tự động tính toán các đặc trưng
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI(
    title="Student Dropout Prediction API",
    description="API dự đoán khả năng bỏ học của học viên dựa trên hoạt động học tập",
    version="1.0.0"
)

# Load model
MODEL_PATH = "short_term_inactive_next7days_logreg.pkl"
try:
    model_bundle = joblib.load(MODEL_PATH)
    # Model được lưu dưới dạng dictionary {"pipeline": ..., "feature_cols": ...}
    model = model_bundle["pipeline"] if isinstance(model_bundle, dict) else model_bundle
    model_feature_cols = model_bundle.get("feature_cols") if isinstance(model_bundle, dict) else None
    logger.info(f"Đã load model từ {MODEL_PATH}")
    if model_feature_cols:
        logger.info(f"Model features: {model_feature_cols}")
except Exception as e:
    logger.error(f"Không thể load model: {e}")
    model = None
    model_feature_cols = None

# Cấu hình
WINDOW_DAYS = 14
HALF_WINDOW = 7


class VLEActivity(BaseModel):
    """Dữ liệu hoạt động VLE (Virtual Learning Environment) của học viên"""
    date: int = Field(..., description="Ngày hoạt động (tính từ ngày đăng ký, VD: 0, 1, 2, ...)")
    sum_click: int = Field(..., description="Tổng số clicks trong ngày đó")


class AssessmentSubmission(BaseModel):
    """Dữ liệu nộp bài đánh giá của học viên"""
    date_submitted: int = Field(..., description="Ngày nộp bài (tính từ ngày đăng ký)")
    score: float = Field(..., description="Điểm số (0-100)")


class StudentData(BaseModel):
    """Dữ liệu đầu vào của học viên để dự đoán"""
    student_id: str = Field(..., description="ID của học viên")
    days_since_registration: int = Field(..., description="Số ngày kể từ khi đăng ký (VD: 30, 60, 90)")
    vle_activities: List[VLEActivity] = Field(..., description="Danh sách hoạt động VLE")
    assessment_submissions: Optional[List[AssessmentSubmission]] = Field(
        default=[], 
        description="Danh sách bài đánh giá đã nộp"
    )


class StudentFeatures(BaseModel):
    """Features đã được tính toán sẵn từ BE (cho batch processing)"""
    student_id: str = Field(..., description="ID của học viên")
    days_elapsed_since_reg: float = Field(..., description="Số ngày kể từ khi đăng ký")
    clicks_per_day_total: float = Field(..., description="Clicks trung bình mỗi ngày (tổng)")
    active_ratio_total: float = Field(..., description="Tỷ lệ ngày hoạt động (tổng)")
    avg_clicks_per_active_day_total: float = Field(..., description="Clicks TB mỗi ngày hoạt động")
    days_since_last_active: float = Field(..., description="Số ngày kể từ lần hoạt động cuối")
    clicks_last_14_days: float = Field(..., description="Tổng clicks trong 14 ngày gần nhất")
    active_days_14: float = Field(..., description="Số ngày hoạt động trong 14 ngày")
    clicks_per_day_14: float = Field(..., description="Clicks TB mỗi ngày (14 ngày)")
    active_ratio_14: float = Field(..., description="Tỷ lệ ngày hoạt động (14 ngày)")
    clicks_last_7_days: float = Field(..., description="Clicks trong 7 ngày gần nhất")
    clicks_0_7: float = Field(..., description="Clicks trong nửa đầu cửa sổ 14 ngày")
    clicks_8_14: float = Field(..., description="Clicks trong nửa sau cửa sổ 14 ngày")
    trend_click_14: float = Field(..., description="Xu hướng clicks (8-14 trừ 0-7)")
    ratio_click_14: float = Field(..., description="Tỷ lệ clicks giữa 2 nửa cửa sổ")
    inactivity_streak_14: float = Field(..., description="Số ngày không hoạt động liên tiếp")


class PredictionResponse(BaseModel):
    """Kết quả dự đoán"""
    student_id: str
    dropout_probability: float = Field(..., description="Xác suất bỏ học (0-1)")
    risk_level: str = Field(..., description="Mức độ rủi ro: LOW, MEDIUM, HIGH")
    features_used: dict = Field(..., description="Các đặc trưng đã tính toán")


class BatchPredictionResponse(BaseModel):
    """Kết quả dự đoán cho batch (không bao gồm features_used để giảm payload)"""
    student_id: str
    dropout_probability: float = Field(..., description="Xác suất bỏ học (0-1)")
    risk_level: str = Field(..., description="Mức độ rủi ro: LOW, MEDIUM, HIGH")


def compute_inactivity_streak(days_list: List[int], start_day: int, end_day: int) -> int:
    """Tính số ngày không hoạt động liên tiếp tính từ end_day về start_day"""
    if not days_list:
        return end_day - start_day + 1
    active = set(days_list)
    streak, d = 0, end_day
    while d >= start_day and d not in active:
        streak += 1
        d -= 1
    return streak


def build_features_for_prediction(
    student_data: StudentData,
    window_days: int = WINDOW_DAYS,
    half_window: int = HALF_WINDOW
) -> dict:
    """
    Tính toán các đặc trưng từ dữ liệu thô của học viên
    
    Args:
        student_data: Dữ liệu học viên
        window_days: Số ngày trong cửa sổ quan sát (mặc định 14)
        half_window: Nửa cửa sổ (mặc định 7)
    
    Returns:
        Dictionary chứa các đặc trưng đã tính toán
    """
    cutoff = student_data.days_since_registration
    w_start = max(0, cutoff - (window_days - 1))
    w_end = cutoff
    
    # Chuyển VLE activities thành DataFrame
    vle_df = pd.DataFrame([
        {"days_since_reg": act.date, "sum_click": act.sum_click} 
        for act in student_data.vle_activities
    ])
    
    # Lọc dữ liệu theo cutoff
    vle_cum = vle_df[vle_df["days_since_reg"] <= cutoff].copy() if len(vle_df) > 0 else pd.DataFrame()
    vle_win = vle_cum[vle_cum["days_since_reg"] >= w_start].copy() if len(vle_cum) > 0 else pd.DataFrame()
    
    # === Tính các đặc trưng tích lũy (cumulative) ===
    if len(vle_cum) > 0:
        total_clicks = vle_cum["sum_click"].sum()
        active_days_total = vle_cum["days_since_reg"].nunique()
        last_active = vle_cum["days_since_reg"].max()
        clicks_per_day_total = total_clicks / max(cutoff, 1)
        active_ratio_total = active_days_total / max(cutoff, 1)
        days_since_last_active = cutoff - last_active
        avg_clicks_per_active_day_total = total_clicks / max(active_days_total, 1)
    else:
        total_clicks = 0
        active_days_total = 0
        last_active = 0
        clicks_per_day_total = 0
        active_ratio_total = 0
        days_since_last_active = cutoff
        avg_clicks_per_active_day_total = 0
    
    # === Tính các đặc trưng trong cửa sổ 14 ngày ===
    if len(vle_win) > 0:
        clicks_last_14_days = vle_win["sum_click"].sum()
        active_days_14 = vle_win["days_since_reg"].nunique()
        clicks_per_day_14 = clicks_last_14_days / window_days
        active_ratio_14 = active_days_14 / window_days
        
        # Tách 2 nửa cửa sổ
        first_end = min(w_end, w_start + (half_window - 1))
        second_start = min(w_end, first_end + 1)
        
        clicks_0_7 = vle_win[
            (vle_win["days_since_reg"] >= w_start) & 
            (vle_win["days_since_reg"] <= first_end)
        ]["sum_click"].sum()
        
        clicks_8_14 = vle_win[
            (vle_win["days_since_reg"] >= second_start) & 
            (vle_win["days_since_reg"] <= w_end)
        ]["sum_click"].sum()
        
        # Tính inactivity streak
        active_days_list = sorted(vle_win["days_since_reg"].unique().tolist())
        inactivity_streak_14 = compute_inactivity_streak(active_days_list, w_start, w_end)
    else:
        clicks_last_14_days = 0
        active_days_14 = 0
        clicks_per_day_14 = 0
        active_ratio_14 = 0
        clicks_0_7 = 0
        clicks_8_14 = 0
        inactivity_streak_14 = window_days
    
    # === Tính clicks trong 7 ngày gần nhất ===
    if len(vle_cum) > 0:
        clicks_last_7_days = vle_cum[vle_cum["days_since_reg"] > (cutoff - 7)]["sum_click"].sum()
    else:
        clicks_last_7_days = 0
    
    # === Tính trend và ratio ===
    trend_click_14 = clicks_8_14 - clicks_0_7
    ratio_click_14 = (clicks_8_14 + 1) / (clicks_0_7 + 1)
    
    # === Tính các đặc trưng về assessment (nếu có) ===
    # Commented out vì model hiện tại không dùng
    """
    if student_data.assessment_submissions:
        ass_df = pd.DataFrame([
            {"days_since_reg": sub.date_submitted, "score": sub.score}
            for sub in student_data.assessment_submissions
        ])
        ass_cum = ass_df[ass_df["days_since_reg"] <= cutoff]
        
        if len(ass_cum) > 0:
            num_assessments = len(ass_cum)
            avg_score = ass_cum["score"].mean()
            pass_count = (ass_cum["score"] >= 40).sum()
            last_score = ass_cum.iloc[-1]["score"]
            
            ass_win = ass_cum[ass_cum["days_since_reg"] >= w_start]
            num_assessments_14 = len(ass_win) if len(ass_win) > 0 else 0
            avg_score_14 = ass_win["score"].mean() if len(ass_win) > 0 else 0
        else:
            num_assessments = avg_score = pass_count = last_score = 0
            num_assessments_14 = avg_score_14 = 0
    else:
        num_assessments = avg_score = pass_count = last_score = 0
        num_assessments_14 = avg_score_14 = 0
    """
    
    # Trả về các đặc trưng theo thứ tự model yêu cầu
    features = {
        "days_elapsed_since_reg": cutoff,
        "clicks_per_day_total": clicks_per_day_total,
        "active_ratio_total": active_ratio_total,
        "avg_clicks_per_active_day_total": avg_clicks_per_active_day_total,
        "days_since_last_active": days_since_last_active,
        "clicks_last_14_days": clicks_last_14_days,
        "active_days_14": active_days_14,
        "clicks_per_day_14": clicks_per_day_14,
        "active_ratio_14": active_ratio_14,
        "clicks_last_7_days": clicks_last_7_days,
        "clicks_0_7": clicks_0_7,
        "clicks_8_14": clicks_8_14,
        "trend_click_14": trend_click_14,
        "ratio_click_14": ratio_click_14,
        "inactivity_streak_14": inactivity_streak_14,
    }
    
    return features


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Student Dropout Prediction API is running",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_dropout(student_data: StudentData):
    """
    Dự đoán khả năng bỏ học của học viên
    
    Args:
        student_data: Dữ liệu học viên bao gồm:
            - student_id: ID học viên
            - days_since_registration: Số ngày kể từ khi đăng ký
            - vle_activities: Danh sách hoạt động VLE (ngày và số clicks)
            - assessment_submissions: Danh sách bài đánh giá đã nộp (optional)
    
    Returns:
        PredictionResponse với xác suất bỏ học và mức độ rủi ro
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    try:
        # Tính toán các đặc trưng
        features = build_features_for_prediction(student_data)
        logger.info(f"Đã tính toán features cho student {student_data.student_id}")
        
        # Chuyển features thành DataFrame với đúng thứ tự cột
        feature_cols = model_feature_cols or [
            "days_elapsed_since_reg",
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
        ]
        
        X = pd.DataFrame([features])[feature_cols].fillna(0)
        
        # Dự đoán với pipeline object
        dropout_proba = model.predict_proba(X)[0, 1]
        
        # Xác định mức độ rủi ro
        if dropout_proba < 0.3:
            risk_level = "LOW"
        elif dropout_proba < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        logger.info(
            f"Dự đoán cho student {student_data.student_id}: "
            f"probability={dropout_proba:.3f}, risk={risk_level}"
        )
        
        return PredictionResponse(
            student_id=student_data.student_id,
            dropout_probability=float(dropout_proba),
            risk_level=risk_level,
            features_used=features
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán cho student {student_data.student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")


@app.post("/predict/batch")
def predict_dropout_batch(students: List[StudentData]):
    """
    Dự đoán khả năng bỏ học cho nhiều học viên cùng lúc
    
    Args:
        students: Danh sách dữ liệu học viên
    
    Returns:
        Danh sách kết quả dự đoán
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    results = []
    for student_data in students:
        try:
            result = predict_dropout(student_data)
            results.append(result)
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán cho student {student_data.student_id}: {e}")
            # Tiếp tục xử lý các học viên khác
            results.append({
                "student_id": student_data.student_id,
                "error": str(e)
            })
    
    return results


@app.post("/predict/features", response_model=List[BatchPredictionResponse])
def predict_with_precomputed_features(students_features: List[StudentFeatures]):
    """
    Dự đoán khả năng bỏ học với features đã được tính toán sẵn (tối ưu cho batch/cronjob)
    
    Endpoint này dành cho trường hợp BE đã tính toán features trước.
    Phù hợp cho batch processing với hàng nghìn học viên.
    
    Args:
        students_features: Danh sách features đã tính toán cho từng học viên
    
    Returns:
        Danh sách kết quả dự đoán (không bao gồm features_used để giảm payload)
    
    Example:
        ```json
        [
          {
            "student_id": "12345",
            "days_elapsed_since_reg": 60,
            "clicks_per_day_total": 2.5,
            "active_ratio_total": 0.4,
            ...
          }
        ]
        ```
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    logger.info(f"📥 Nhận request predict cho {len(students_features)} học viên")
    
    try:
        # Chuyển features thành DataFrame
        feature_cols = model_feature_cols or [
            "days_elapsed_since_reg",
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
        ]
        
        # Tạo DataFrame từ list of StudentFeatures
        features_list = []
        student_ids = []
        for sf in students_features:
            student_ids.append(sf.student_id)
            features_list.append({
                "days_elapsed_since_reg": sf.days_elapsed_since_reg,
                "clicks_per_day_total": sf.clicks_per_day_total,
                "active_ratio_total": sf.active_ratio_total,
                "avg_clicks_per_active_day_total": sf.avg_clicks_per_active_day_total,
                "days_since_last_active": sf.days_since_last_active,
                "clicks_last_14_days": sf.clicks_last_14_days,
                "active_days_14": sf.active_days_14,
                "clicks_per_day_14": sf.clicks_per_day_14,
                "active_ratio_14": sf.active_ratio_14,
                "clicks_last_7_days": sf.clicks_last_7_days,
                "clicks_0_7": sf.clicks_0_7,
                "clicks_8_14": sf.clicks_8_14,
                "trend_click_14": sf.trend_click_14,
                "ratio_click_14": sf.ratio_click_14,
                "inactivity_streak_14": sf.inactivity_streak_14,
            })
        
        X = pd.DataFrame(features_list)[feature_cols].fillna(0)
        
        # Log features của từng student
        for idx, (student_id, features) in enumerate(zip(student_ids, features_list)):
            logger.info(f"\n👤 Student {student_id}:")
            logger.info(f"   days_elapsed: {features['days_elapsed_since_reg']}")
            logger.info(f"   clicks_14d: {features['clicks_last_14_days']}, active_days_14: {features['active_days_14']}")
            logger.info(f"   active_ratio_14: {features['active_ratio_14']:.2%}")
            logger.info(f"   inactivity_streak: {features['inactivity_streak_14']}")
            logger.info(f"   trend: {features['trend_click_14']}, ratio: {features['ratio_click_14']:.2f}")
        
        # Dự đoán batch
        dropout_probas = model.predict_proba(X)[:, 1]
        
        logger.info("\n🎯 Kết quả prediction:")
        
        # Tạo kết quả
        results = []
        for student_id, proba in zip(student_ids, dropout_probas):
            if proba < 0.35:
                risk_level = "LOW"
            elif proba < 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            logger.info(f"   {student_id}: {proba:.1%} ({risk_level})")
            
            results.append(BatchPredictionResponse(
                student_id=student_id,
                dropout_probability=float(proba),
                risk_level=risk_level
            ))
        
        logger.info(f"\n✅ Batch prediction hoàn thành cho {len(students_features)} học viên")
        return results
        
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán batch: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán batch: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
