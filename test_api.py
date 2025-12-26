"""
Script test API dự đoán khả năng bỏ học
Chạy sau khi đã start API server
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction():
    """Test dự đoán cho 1 học viên"""
    print("=== Testing Single Prediction ===")
    
    # Case 1: Học viên hoạt động tốt (low risk)
    student_good = {
        "student_id": "GOOD_001",
        "days_since_registration": 60,
        "vle_activities": [
            {"date": i, "sum_click": 80 + (i % 20)} 
            for i in range(0, 60, 2)  # Hoạt động đều đặn
        ],
        "assessment_submissions": [
            {"date_submitted": 15, "score": 85},
            {"date_submitted": 30, "score": 78},
            {"date_submitted": 45, "score": 82}
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=student_good)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Student: {result['student_id']}")
        print(f"Dropout Probability: {result['dropout_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Features used: {json.dumps(result['features_used'], indent=2)}\n")
    else:
        print(f"Error: {response.text}\n")
    
    # Case 2: Học viên có dấu hiệu bỏ học (high risk)
    student_risk = {
        "student_id": "RISK_001",
        "days_since_registration": 60,
        "vle_activities": [
            {"date": i, "sum_click": max(5, 100 - i * 2)} 
            for i in range(0, 30, 3)  # Hoạt động giảm dần và thưa
        ],
        "assessment_submissions": [
            {"date_submitted": 20, "score": 35}  # Điểm thấp
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=student_risk)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Student: {result['student_id']}")
        print(f"Dropout Probability: {result['dropout_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Key features:")
        features = result['features_used']
        print(f"  - Days since last active: {features['days_since_last_active']}")
        print(f"  - Active ratio (total): {features['active_ratio_total']:.2%}")
        print(f"  - Active ratio (14d): {features['active_ratio_14']:.2%}")
        print(f"  - Inactivity streak: {features['inactivity_streak_14']} days\n")
    else:
        print(f"Error: {response.text}\n")


def test_batch_prediction():
    """Test dự đoán cho nhiều học viên"""
    print("=== Testing Batch Prediction ===")
    
    students = [
        {
            "student_id": f"STUDENT_{i:03d}",
            "days_since_registration": 30,
            "vle_activities": [
                {"date": j, "sum_click": 50 + (i * 10) - (j % 5)} 
                for j in range(0, 30, 2)
            ],
            "assessment_submissions": []
        }
        for i in range(1, 4)
    ]
    
    response = requests.post(f"{API_URL}/predict/batch", json=students)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        for result in results:
            if 'dropout_probability' in result:
                print(f"  {result['student_id']}: "
                      f"{result['dropout_probability']:.2%} - "
                      f"{result['risk_level']}")
            else:
                print(f"  {result['student_id']}: ERROR - {result.get('error', 'Unknown')}")
    else:
        print(f"Error: {response.text}")
    print()


def test_batch_with_precomputed_features():
    """Test dự đoán với features đã tính sẵn (tối ưu cho cronjob)"""
    print("=== Testing Batch Prediction with Precomputed Features ===")
    
    # Giả sử BE đã tính toán features
    students_features = [
        {
            "student_id": "PRECOMP_001",
            "days_elapsed_since_reg": 60,
            "clicks_per_day_total": 5.5,
            "active_ratio_total": 0.6,
            "avg_clicks_per_active_day_total": 91.67,
            "days_since_last_active": 2,
            "clicks_last_14_days": 200,
            "active_days_14": 10,
            "clicks_per_day_14": 14.29,
            "active_ratio_14": 0.71,
            "clicks_last_7_days": 120,
            "clicks_0_7": 110,
            "clicks_8_14": 90,
            "trend_click_14": -20,
            "ratio_click_14": 0.82,
            "inactivity_streak_14": 2
        },
        {
            "student_id": "PRECOMP_002",
            "days_elapsed_since_reg": 60,
            "clicks_per_day_total": 1.2,
            "active_ratio_total": 0.2,
            "avg_clicks_per_active_day_total": 36.0,
            "days_since_last_active": 15,
            "clicks_last_14_days": 30,
            "active_days_14": 3,
            "clicks_per_day_14": 2.14,
            "active_ratio_14": 0.21,
            "clicks_last_7_days": 10,
            "clicks_0_7": 20,
            "clicks_8_14": 10,
            "trend_click_14": -10,
            "ratio_click_14": 0.48,
            "inactivity_streak_14": 10
        },
        {
            "student_id": "PRECOMP_003",
            "days_elapsed_since_reg": 90,
            "clicks_per_day_total": 8.0,
            "active_ratio_total": 0.75,
            "avg_clicks_per_active_day_total": 106.67,
            "days_since_last_active": 0,
            "clicks_last_14_days": 300,
            "active_days_14": 12,
            "clicks_per_day_14": 21.43,
            "active_ratio_14": 0.86,
            "clicks_last_7_days": 180,
            "clicks_0_7": 140,
            "clicks_8_14": 160,
            "trend_click_14": 20,
            "ratio_click_14": 1.14,
            "inactivity_streak_14": 0
        }
    ]
    
    response = requests.post(f"{API_URL}/predict/features", json=students_features)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"Processed {len(results)} students:")
        for result in results:
            print(f"  {result['student_id']}: "
                  f"{result['dropout_probability']:.2%} - "
                  f"{result['risk_level']}")
        
        # Phân tích kết quả
        print("\n📊 Analysis:")
        high_risk = [r for r in results if r['risk_level'] == 'HIGH']
        medium_risk = [r for r in results if r['risk_level'] == 'MEDIUM']
        low_risk = [r for r in results if r['risk_level'] == 'LOW']
        
        print(f"  HIGH risk: {len(high_risk)} students")
        print(f"  MEDIUM risk: {len(medium_risk)} students")
        print(f"  LOW risk: {len(low_risk)} students")
    else:
        print(f"Error: {response.text}")
    print()


def test_edge_cases():
    """Test các trường hợp đặc biệt"""
    print("=== Testing Edge Cases ===")
    
    # Case 1: Học viên không có hoạt động nào
    print("1. No VLE activities:")
    student_no_activity = {
        "student_id": "NO_ACT_001",
        "days_since_registration": 30,
        "vle_activities": [],
        "assessment_submissions": []
    }
    
    response = requests.post(f"{API_URL}/predict", json=student_no_activity)
    if response.status_code == 200:
        result = response.json()
        print(f"   Dropout Probability: {result['dropout_probability']:.2%}")
        print(f"   Risk Level: {result['risk_level']}")
    else:
        print(f"   Error: {response.text}")
    
    # Case 2: Học viên mới (chỉ 7 ngày)
    print("\n2. New student (7 days):")
    student_new = {
        "student_id": "NEW_001",
        "days_since_registration": 7,
        "vle_activities": [
            {"date": i, "sum_click": 100} for i in range(7)
        ],
        "assessment_submissions": []
    }
    
    response = requests.post(f"{API_URL}/predict", json=student_new)
    if response.status_code == 200:
        result = response.json()
        print(f"   Dropout Probability: {result['dropout_probability']:.2%}")
        print(f"   Risk Level: {result['risk_level']}")
    else:
        print(f"   Error: {response.text}")
    
    # Case 3: Học viên đã học lâu (180 ngày)
    print("\n3. Long-time student (180 days):")
    student_long = {
        "student_id": "LONG_001",
        "days_since_registration": 180,
        "vle_activities": [
            {"date": i, "sum_click": 60 + (i % 30)} 
            for i in range(0, 180, 3)
        ],
        "assessment_submissions": [
            {"date_submitted": i * 30, "score": 70 + (i * 5)} 
            for i in range(1, 6)
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=student_long)
    if response.status_code == 200:
        result = response.json()
        print(f"   Dropout Probability: {result['dropout_probability']:.2%}")
        print(f"   Risk Level: {result['risk_level']}")
    else:
        print(f"   Error: {response.text}")
    
    print()


def main():
    """Chạy tất cả các test"""
    try:
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_batch_with_precomputed_features()
        test_edge_cases()
        
        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Không thể kết nối đến API server!")
        print("Hãy chạy API server trước: python predict_api.py")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
