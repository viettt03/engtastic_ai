# API Dự Đoán Khả Năng Bỏ Học

API này cho phép backend gọi để dự đoán khả năng bỏ học của học viên. Backend chỉ cần truyền dữ liệu thô (hoạt động VLE, bài đánh giá), API sẽ tự động tính toán các đặc trưng và trả về kết quả.

## Cài Đặt

```bash
pip install -r api_requirements.txt
```

## Chạy API

```bash
python predict_api.py
```

Hoặc với uvicorn:

```bash
uvicorn predict_api:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ chạy tại: `http://localhost:8000`

Swagger UI docs: `http://localhost:8000/docs`

## Endpoints

### 1. Health Check

```
GET /
```

### 2. Dự đoán cho 1 học viên (Realtime)

```
POST /predict
```

**Use case:** Dự đoán realtime khi xem profile học viên

**Request Body:**

```json
{
  "student_id": "12345",
  "days_since_registration": 60,
  "vle_activities": [
    { "date": 1, "sum_click": 50 },
    { "date": 2, "sum_click": 30 },
    { "date": 5, "sum_click": 80 }
  ],
  "assessment_submissions": [
    { "date_submitted": 10, "score": 75 },
    { "date_submitted": 25, "score": 65 }
  ]
}
```

**Response:**

```json
{
  "student_id": "12345",
  "dropout_probability": 0.35,
  "risk_level": "MEDIUM",
  "features_used": {
    "days_elapsed_since_reg": 60,
    "clicks_per_day_total": 2.67,
    "active_ratio_total": 0.05
  }
}
```

### 3. Dự đoán với raw data (Batch - API tự tính features)

```
POST /predict/batch
```

**Use case:** Batch processing nhỏ (< 100 học viên)

**Request Body:** Array of student data (same format as `/predict`)

### 4. Dự đoán với features đã tính (Batch - Tối ưu cho cronjob) ⭐

```
POST /predict/features
```

**Use case:** Cronjob xử lý hàng nghìn học viên

**Ưu điểm:**

- Hiệu năng cao hơn 10-20 lần
- Payload nhỏ hơn (15 features vs hàng trăm/nghìn VLE records)
- Network overhead thấp
- Có thể cache features trong DB

**Request Body:**

```json
[
  {
    "student_id": "12345",
    "days_elapsed_since_reg": 60,
    "clicks_per_day_total": 2.67,
    "active_ratio_total": 0.05,
    "avg_clicks_per_active_day_total": 53.4,
    "days_since_last_active": 5,
    "clicks_last_14_days": 150,
    "active_days_14": 8,
    "clicks_per_day_14": 10.71,
    "active_ratio_14": 0.57,
    "clicks_last_7_days": 80,
    "clicks_0_7": 90,
    "clicks_8_14": 60,
    "trend_click_14": -30,
    "ratio_click_14": 0.67,
    "inactivity_streak_14": 3
  }
]
```

**Response:**

```json
[
  {
    "student_id": "12345",
    "dropout_probability": 0.35,
    "risk_level": "MEDIUM"
  }
]
```

**Note:** Response không bao gồm `features_used` để giảm payload khi xử lý batch lớn.

**Risk Levels:**

- `LOW`: xác suất < 0.3
- `MEDIUM`: xác suất 0.3 - 0.6
- `HIGH`: xác suất > 0.6

### 3. Dự đoán cho nhiều học viên

```
POST /predict/batch
```

**Request Body:** Array of student data (same format as single prediction)

## Cấu Trúc Dữ Liệu Đầu Vào

### VLE Activity

- `date`: Số ngày kể từ khi đăng ký (0, 1, 2, ...)
- `sum_click`: Tổng số clicks trong ngày đó

**Lưu ý:** `date` được tính từ ngày đăng ký, không phải ngày thực tế. VD:

- Học viên đăng ký ngày 1/1/2024
- Hoạt động ngày 5/1/2024 → `date = 4`
- Hoạt động ngày 15/1/2024 → `date = 14`

### Assessment Submission (Optional)

- `date_submitted`: Ngày nộp bài (tính từ ngày đăng ký)
- `score`: Điểm số (0-100)

### Days Since Registration

- Số ngày kể từ khi đăng ký đến thời điểm cần dự đoán
- VD: Muốn dự đoán sau 30 ngày → `days_since_registration = 30`
- Nên dự đoán tại các mốc: 7, 14, 30, 60, 90, 120, 150, 180 ngày

## Ví Dụ Sử Dụng với Python

```python
import requests

# Dữ liệu học viên
student_data = {
    "student_id": "67890",
    "days_since_registration": 30,
    "vle_activities": [
        {"date": 0, "sum_click": 100},
        {"date": 1, "sum_click": 80},
        {"date": 2, "sum_click": 50},
        {"date": 5, "sum_click": 120},
        {"date": 7, "sum_click": 90},
        {"date": 10, "sum_click": 60},
        {"date": 15, "sum_click": 40},
        {"date": 20, "sum_click": 30},
        {"date": 25, "sum_click": 20},
        {"date": 28, "sum_click": 10},
    ],
    "assessment_submissions": []
}

# Gọi API
response = requests.post(
    "http://localhost:8000/predict",
    json=student_data
)

result = response.json()
print(f"Student ID: {result['student_id']}")
print(f"Dropout Probability: {result['dropout_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

## Ví Dụ với cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "12345",
    "days_since_registration": 60,
    "vle_activities": [
      {"date": 1, "sum_click": 50},
      {"date": 2, "sum_click": 30}
    ],
    "assessment_submissions": []
  }'
```

## Ví Dụ với JavaScript/TypeScript

```typescript
const studentData = {
  student_id: "12345",
  days_since_registration: 60,
  vle_activities: [
    { date: 1, sum_click: 50 },
    { date: 2, sum_click: 30 },
  ],
  assessment_submissions: [],
};

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(studentData),
});

const result = await response.json();
console.log(result);
```

## Đặc Trưng Được Tính Toán

API tự động tính các đặc trưng sau từ dữ liệu thô:

1. **Cumulative Features** (Tích lũy từ đầu):

   - `clicks_per_day_total`: Clicks trung bình mỗi ngày
   - `active_ratio_total`: Tỷ lệ ngày hoạt động
   - `avg_clicks_per_active_day_total`: Clicks TB/ngày hoạt động
   - `days_since_last_active`: Số ngày kể từ lần hoạt động cuối

2. **Window Features** (14 ngày gần nhất):

   - `clicks_last_14_days`: Tổng clicks trong 14 ngày
   - `active_days_14`: Số ngày hoạt động trong 14 ngày
   - `clicks_per_day_14`: Clicks TB/ngày trong 14 ngày
   - `active_ratio_14`: Tỷ lệ ngày hoạt động trong 14 ngày
   - `inactivity_streak_14`: Số ngày không hoạt động liên tiếp

3. **Trend Features** (Xu hướng):
   - `clicks_0_7`: Clicks trong 7 ngày đầu của cửa sổ
   - `clicks_8_14`: Clicks trong 7 ngày sau của cửa sổ
   - `trend_click_14`: Xu hướng tăng/giảm
   - `ratio_click_14`: Tỷ lệ giữa 2 nửa cửa sổ
   - `clicks_last_7_days`: Clicks trong 7 ngày gần nhất

## Lưu Ý Quan Trọng

1. **Model File**: Đảm bảo file `dropout_model_relative.pkl` tồn tại trong cùng thư mục
2. **Dữ liệu đầu vào**:
   - Tất cả `date` đều tính từ ngày đăng ký (relative date)
   - Chỉ truyền dữ liệu <= `days_since_registration`
3. **Performance**: Với batch prediction, API xử lý tuần tự từng học viên
4. **Error Handling**: API trả về lỗi chi tiết khi có vấn đề

## Deploy với Docker

Xem file `Dockerfile` và `docker-compose.yml` để deploy production.

## Monitoring & Logging

API tự động log các thông tin:

- Request prediction
- Features calculated
- Prediction results
- Errors

Logs có thể được forward đến monitoring system (ELK, Datadog, etc.)
