# 🚀 Deployment Guide - Dropout Predictor AI Service

## 📋 Tổng quan
Service FastAPI này phục vụ mô hình dự đoán dropout của học sinh. Backend NestJS trên server có thể gọi API này để nhận prediction.

## 🏗️ Cấu trúc
```
engtastic_ai/
├── serve_dropout_model.py    # FastAPI service
├── requirements.txt           # Python dependencies
├── artifacts/
│   └── dropout_early21.joblib # Trained model (REQUIRED)
├── Dockerfile                 # Docker image config
├── docker-compose.yml         # Docker orchestration
└── DEPLOYMENT.md             # This file
```

## ⚙️ Cài đặt & Deploy

### 1. Chuẩn bị trên Server

```bash
# Tạo network chung (nếu chưa có)
docker network create lms

# Upload toàn bộ thư mục engtastic_ai lên server
# Đảm bảo có file artifacts/dropout_early21.joblib
```

### 2. Build & Start Service

```bash
cd engtastic_ai

# Build image
docker-compose build

# Start service (detached mode)
docker-compose up -d

# Xem logs
docker-compose logs -f
```

### 3. Kiểm tra Health

```bash
# Từ server
curl http://localhost:8001/health

# Từ container khác trong network lms
curl http://dropout-predictor:8001/health
```

Expected response:
```json
{
  "status": "ok",
  "model_name": "LightGBM",
  "module": "BBB",
  "presentation": "2013J",
  "early_days": 21,
  "features": [...]
}
```

## 🌐 Nginx Configuration

Đã được cấu hình trong `nginx.conf`:

```nginx
# AI Service accessible at https://engtastic.app/ai/
location ^~ /ai/ {
    proxy_pass http://dropout-predictor:8001/;
}
```

**Restart Nginx** sau khi update config:
```bash
docker exec nginx-container nginx -s reload
# hoặc
docker-compose restart nginx
```

## 🔌 Sử dụng từ Backend (NestJS)

### Internal Call (trong Docker network)
```typescript
// Từ container khác trong network lms
const response = await axios.post('http://dropout-predictor:8001/predict', {
  students: [{
    id_student: 123,
    total_clicks: 450,
    active_days: 15,
    // ... other features
    gender: "M",
    age_band: "0-35"
  }],
  threshold: 0.5
});
```

### External Call (qua Nginx)
```typescript
// Từ bên ngoài hoặc thông qua domain
const response = await axios.post('https://engtastic.app/ai/predict', {
  students: [...],
  threshold: 0.5
});
```

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Predict Dropout
```bash
POST /predict

Body:
{
  "students": [
    {
      "id_student": 123,
      "total_clicks": 450.0,
      "active_days": 15.0,
      "avg_clicks_per_day": 30.0,
      "avg_clicks_per_active_day": 30.0,
      "clicks_0_7": 210.0,
      "clicks_8_14": 210.0,
      "clicks_15_21": 30.0,
      "trend_click": 0.5,
      "ratio_click": 1.0,
      "num_assessments": 3.0,
      "avg_score": 75.0,
      "max_score": 90.0,
      "min_score": 60.0,
      "score_std": 15.0,
      "last_score": 75.0,
      "pass_rate": 0.67,
      "reg_day": -5.0,
      "registered_before_start": 1.0,
      "days_since_last_login": 2.0,
      "inactivity_streak": 1.0,
      "gender": "M",
      "age_band": "0-35"
    }
  ],
  "threshold": 0.5
}

Response:
{
  "model_name": "LightGBM",
  "module": "BBB",
  "presentation": "2013J",
  "early_days": 21,
  "results": [
    {
      "id_student": 123,
      "dropout_probability": 0.35,
      "dropout_prediction": 0
    }
  ]
}
```

## 🔧 Management Commands

```bash
# Stop service
docker-compose down

# Restart service
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build

# View logs
docker-compose logs -f dropout-predictor

# Remove everything
docker-compose down -v
docker rmi engtastic-ai-dropout-predictor
```

## 🐛 Troubleshooting

### Service không start được
```bash
# Check logs
docker-compose logs dropout-predictor

# Kiểm tra network
docker network inspect lms

# Verify artifacts exists
docker exec dropout-predictor ls -la /app/artifacts/
```

### Model file không tìm thấy
```bash
# Đảm bảo file artifacts/dropout_early21.joblib tồn tại
ls -la artifacts/

# Build lại với --no-cache
docker-compose build --no-cache
```

### Connection refused từ backend
```bash
# Kiểm tra container đang chạy
docker ps | grep dropout

# Test connectivity từ backend container
docker exec lms-backend-lms-be-1 curl http://dropout-predictor:8001/health

# Kiểm tra network
docker exec lms-backend-lms-be-1 ping dropout-predictor
```

## 📊 Monitoring

```bash
# Resource usage
docker stats dropout-predictor

# Health status
watch -n 5 'curl -s http://localhost:8001/health | jq .'
```

## 🔐 Security Notes

1. Service chỉ expose `127.0.0.1:8001` trên host
2. Access từ bên ngoài phải qua Nginx
3. Trong Docker network, các service khác có thể gọi trực tiếp
4. HTTPS được handle bởi Nginx

## 📝 Update Model

Khi có model mới:
```bash
# 1. Upload file mới vào artifacts/
# 2. Restart service
docker-compose restart

# hoặc rebuild
docker-compose up -d --build
```

---

## ✅ Checklist Deploy

- [ ] Docker network `lms` đã được tạo
- [ ] File `artifacts/dropout_early21.joblib` tồn tại
- [ ] `requirements.txt` đầy đủ dependencies
- [ ] Build image thành công
- [ ] Service start và healthy
- [ ] Health endpoint response OK
- [ ] Nginx config đã update
- [ ] Nginx đã reload
- [ ] Backend có thể connect đến service
- [ ] Test predict endpoint thành công
