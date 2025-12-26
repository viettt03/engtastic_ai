# Hướng Dẫn Deploy API Dropout Prediction

## 📋 Tổng Quan Kiến Trúc

```
┌─────────────────┐
│   Nginx (443)   │  ← HTTPS engtastic.app/ai/*
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│  Docker Container       │
│  dropout-predictor:8000 │
│  - FastAPI App          │
│  - Model (130MB)        │
│  - CPU: 2 cores         │
│  - RAM: 2GB             │
└─────────────────────────┘
```

## 🚀 Deploy Lần Đầu

### Bước 1: Chuẩn bị files

```bash
cd /path/to/engtastic_ai

# Đảm bảo có các files cần thiết
ls -la dropout_model_relative.pkl  # Model file
ls -la predict_api.py               # API code
ls -la api_requirements.txt         # Dependencies
ls -la Dockerfile                   # Docker config
ls -la docker-compose.yml           # Docker compose
```

### Bước 2: Build Docker image

```bash
# Build image
docker-compose build

# Kiểm tra image đã tạo
docker images | grep engtastic-ai
```

### Bước 3: Start container

```bash
# Start service
docker-compose up -d

# Kiểm tra logs
docker-compose logs -f dropout-predictor

# Kiểm tra health
curl http://localhost:8000/
```

### Bước 4: Cấu hình Nginx

Nginx đã được cấu hình trong `nginx.conf`:

```nginx
location ^~ /ai/ {
    rewrite ^/ai/(.*) /$1 break;
    proxy_pass http://dropout-predictor:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 300s;  # Timeout cho batch processing
}
```

Reload Nginx:

```bash
# Test config
sudo nginx -t

# Reload
sudo systemctl reload nginx
```

### Bước 5: Test API

```bash
# Test từ server (local)
curl http://localhost:8000/

# Test từ bên ngoài (public)
curl https://engtastic.app/ai/

# Test predict endpoint
curl -X POST https://engtastic.app/ai/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test001",
    "days_since_registration": 30,
    "vle_activities": [{"date": 1, "sum_click": 50}],
    "assessment_submissions": []
  }'
```

## 🔄 Update & Redeploy

### Khi update code hoặc model:

```bash
# Pull code mới
git pull

# Rebuild image
docker-compose build

# Restart container (zero downtime nếu dùng blue-green)
docker-compose up -d

# Hoặc force recreate
docker-compose up -d --force-recreate

# Kiểm tra logs
docker-compose logs -f dropout-predictor
```

## 📊 Monitoring & Maintenance

### Xem logs

```bash
# Logs realtime
docker-compose logs -f dropout-predictor

# Logs 100 dòng cuối
docker-compose logs --tail=100 dropout-predictor

# Logs từ 1 giờ trước
docker-compose logs --since 1h dropout-predictor
```

### Kiểm tra tài nguyên

```bash
# CPU, RAM usage
docker stats dropout-predictor

# Disk usage
docker system df
```

### Health check

```bash
# Check container status
docker ps | grep dropout-predictor

# Check health endpoint
curl http://localhost:8000/
```

## 🔧 Troubleshooting

### Container không start

```bash
# Xem logs chi tiết
docker-compose logs dropout-predictor

# Xem events
docker events --filter container=dropout-predictor

# Restart
docker-compose restart dropout-predictor
```

### API chậm hoặc timeout

```bash
# Kiểm tra CPU/RAM
docker stats dropout-predictor

# Tăng resources trong docker-compose.yml
# deploy:
#   resources:
#     limits:
#       cpus: '4'      # Tăng từ 2 lên 4
#       memory: 4G     # Tăng từ 2GB lên 4GB
```

### Model file không load được

```bash
# Kiểm tra file trong container
docker exec -it dropout-predictor ls -lh /app/

# Copy model mới vào container (nếu cần hotfix)
docker cp dropout_model_relative.pkl dropout-predictor:/app/

# Restart container
docker-compose restart dropout-predictor
```

### Nginx 502 Bad Gateway

```bash
# Kiểm tra container có chạy không
docker ps | grep dropout-predictor

# Kiểm tra network
docker network inspect lms

# Kiểm tra container có join đúng network không
docker inspect dropout-predictor | grep -A 10 Networks

# Test connection từ nginx container
docker exec -it <nginx-container> curl http://dropout-predictor:8000/
```

## 🔐 Security Best Practices

### 1. Không expose port ra public

Docker compose đã config `127.0.0.1:8000:8000` - chỉ localhost access được.
Public traffic phải qua Nginx.

### 2. Rate limiting (nếu cần)

Thêm vào nginx config:

```nginx
limit_req_zone $binary_remote_addr zone=ai_limit:10m rate=10r/s;

location ^~ /ai/ {
    limit_req zone=ai_limit burst=20 nodelay;
    # ... existing config
}
```

### 3. API key authentication (nếu cần)

Sửa `predict_api.py`:

```python
from fastapi import Header, HTTPException

API_KEY = "your-secret-key"  # Hoặc đọc từ env

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_dropout(student_data: StudentData):
    # ...
```

## 📈 Performance Optimization

### 1. Caching features trong DB

BE nên cache features đã tính toán:

```sql
CREATE TABLE student_features (
    student_id VARCHAR(50) PRIMARY KEY,
    days_elapsed_since_reg INT,
    clicks_per_day_total FLOAT,
    -- ... các features khác
    calculated_at TIMESTAMP,
    INDEX idx_calculated_at (calculated_at)
);
```

### 2. Batch size optimization

Khi gọi `/predict/features`, chia batch:

```javascript
const BATCH_SIZE = 100;
for (let i = 0; i < students.length; i += BATCH_SIZE) {
  const batch = students.slice(i, i + BATCH_SIZE);
  await callPredictionAPI(batch);
}
```

### 3. Async processing cho cronjob

```javascript
// Xử lý parallel nhưng giới hạn concurrency
const pLimit = require("p-limit");
const limit = pLimit(5); // Max 5 requests đồng thời

const promises = batches.map((batch) => limit(() => callPredictionAPI(batch)));

await Promise.all(promises);
```

## 🎯 Use Cases & Best Practices

### Use Case 1: Realtime prediction (xem profile học viên)

```
User clicks vào profile học viên
  ↓
BE gọi GET /api/students/:id/dropout-risk
  ↓
BE lấy VLE activities từ DB
  ↓
BE gọi POST /ai/predict với raw data
  ↓
API tự tính features và predict
  ↓
Trả về kết quả cho FE hiển thị
```

**Ưu điểm:** Đơn giản, không cần tính features trước  
**Nhược điểm:** Chậm hơn (100-200ms)

### Use Case 2: Daily cronjob (dự đoán cho tất cả học viên)

```
Cronjob chạy lúc 2h sáng hằng ngày
  ↓
BE lấy danh sách tất cả học viên active
  ↓
BE tính features cho từng học viên (có thể song song)
  ↓
BE chia thành batches 100 học viên
  ↓
BE gọi POST /ai/predict/features với features đã tính
  ↓
API chỉ predict (không tính features)
  ↓
BE lưu kết quả vào DB để cache
```

**Ưu điểm:** Nhanh hơn 10-20 lần, giảm tải API  
**Nhược điểm:** BE phải implement logic tính features

**Recommended:** Dùng approach 2 cho cronjob!

## 📝 Monitoring Dashboard

Có thể dùng Prometheus + Grafana để monitor:

```python
# Thêm vào predict_api.py
from prometheus_client import Counter, Histogram, make_asgi_app

prediction_counter = Counter('predictions_total', 'Total predictions', ['endpoint'])
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.post("/predict")
@prediction_duration.time()
def predict_dropout(student_data: StudentData):
    prediction_counter.labels(endpoint='predict').inc()
    # ... existing code

# Mount prometheus endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## 🔄 Backup & Recovery

### Backup model file

```bash
# Backup model
cp dropout_model_relative.pkl dropout_model_relative.pkl.backup.$(date +%Y%m%d)

# Upload to S3 (nếu có)
aws s3 cp dropout_model_relative.pkl s3://your-bucket/models/
```

### Recovery

```bash
# Restore từ backup
cp dropout_model_relative.pkl.backup.20241226 dropout_model_relative.pkl

# Redeploy
docker-compose up -d --force-recreate
```

## 📞 Support & Contact

Nếu có vấn đề, check:

1. Container logs: `docker-compose logs -f`
2. Nginx logs: `/var/log/nginx/error.log`
3. API docs: `https://engtastic.app/ai/docs`
4. Health check: `curl https://engtastic.app/ai/`
