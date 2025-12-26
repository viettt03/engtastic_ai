"""
Code mẫu để BE tính toán features (Java/JavaScript/Python)

Backend có thể implement logic này để tính features trước khi gọi API /predict/features
Điều này tối ưu hơn nhiều cho batch processing (cronjob)
"""

from typing import List, Dict
from datetime import datetime, timedelta


# ============================================================================
# PYTHON VERSION - Có thể dùng trực tiếp hoặc tham khảo để implement sang ngôn ngữ khác
# ============================================================================

class VLEActivity:
    """Hoạt động VLE của học viên"""
    def __init__(self, date: int, sum_click: int):
        self.date = date  # Ngày (tính từ ngày đăng ký)
        self.sum_click = sum_click  # Số clicks


class StudentFeatures:
    """Features đã tính toán"""
    def __init__(self):
        self.days_elapsed_since_reg = 0
        self.clicks_per_day_total = 0.0
        self.active_ratio_total = 0.0
        self.avg_clicks_per_active_day_total = 0.0
        self.days_since_last_active = 0
        self.clicks_last_14_days = 0.0
        self.active_days_14 = 0
        self.clicks_per_day_14 = 0.0
        self.active_ratio_14 = 0.0
        self.clicks_last_7_days = 0.0
        self.clicks_0_7 = 0.0
        self.clicks_8_14 = 0.0
        self.trend_click_14 = 0.0
        self.ratio_click_14 = 0.0
        self.inactivity_streak_14 = 0


def compute_inactivity_streak(active_days: List[int], start_day: int, end_day: int) -> int:
    """Tính số ngày không hoạt động liên tiếp từ end_day về start_day"""
    if not active_days:
        return end_day - start_day + 1
    
    active_set = set(active_days)
    streak = 0
    day = end_day
    
    while day >= start_day and day not in active_set:
        streak += 1
        day -= 1
    
    return streak


def calculate_student_features(
    activities: List[VLEActivity],
    days_since_registration: int,
    window_days: int = 14,
    half_window: int = 7
) -> StudentFeatures:
    """
    Tính toán features cho 1 học viên
    
    Args:
        activities: Danh sách hoạt động VLE (đã sort theo ngày)
        days_since_registration: Số ngày kể từ khi đăng ký (cutoff)
        window_days: Kích thước cửa sổ (mặc định 14)
        half_window: Nửa cửa sổ (mặc định 7)
    
    Returns:
        StudentFeatures đã tính toán
    """
    features = StudentFeatures()
    cutoff = days_since_registration
    w_start = max(0, cutoff - (window_days - 1))
    w_end = cutoff
    
    # Lọc activities theo cutoff
    activities_cum = [a for a in activities if a.date <= cutoff]
    activities_win = [a for a in activities_cum if a.date >= w_start]
    
    # === Tính features tích lũy (cumulative) ===
    if activities_cum:
        total_clicks = sum(a.sum_click for a in activities_cum)
        active_days_list = list(set(a.date for a in activities_cum))
        active_days_total = len(active_days_list)
        last_active = max(a.date for a in activities_cum)
        
        features.clicks_per_day_total = total_clicks / max(cutoff, 1)
        features.active_ratio_total = active_days_total / max(cutoff, 1)
        features.days_since_last_active = cutoff - last_active
        features.avg_clicks_per_active_day_total = total_clicks / max(active_days_total, 1)
    else:
        features.days_since_last_active = cutoff
    
    # === Tính features trong cửa sổ 14 ngày ===
    if activities_win:
        clicks_last_14_days = sum(a.sum_click for a in activities_win)
        active_days_14_list = list(set(a.date for a in activities_win))
        active_days_14 = len(active_days_14_list)
        
        features.clicks_last_14_days = clicks_last_14_days
        features.active_days_14 = active_days_14
        features.clicks_per_day_14 = clicks_last_14_days / window_days
        features.active_ratio_14 = active_days_14 / window_days
        
        # Tính inactivity streak
        features.inactivity_streak_14 = compute_inactivity_streak(
            sorted(active_days_14_list), w_start, w_end
        )
        
        # Tách 2 nửa cửa sổ
        first_end = min(w_end, w_start + (half_window - 1))
        second_start = min(w_end, first_end + 1)
        
        clicks_0_7 = sum(a.sum_click for a in activities_win 
                        if w_start <= a.date <= first_end)
        clicks_8_14 = sum(a.sum_click for a in activities_win 
                         if second_start <= a.date <= w_end)
        
        features.clicks_0_7 = clicks_0_7
        features.clicks_8_14 = clicks_8_14
        features.trend_click_14 = clicks_8_14 - clicks_0_7
        features.ratio_click_14 = (clicks_8_14 + 1) / (clicks_0_7 + 1)
    else:
        features.inactivity_streak_14 = window_days
        features.ratio_click_14 = 1.0
    
    # === Tính clicks trong 7 ngày gần nhất ===
    if activities_cum:
        activities_last_7 = [a for a in activities_cum if a.date > (cutoff - 7)]
        features.clicks_last_7_days = sum(a.sum_click for a in activities_last_7)
    
    features.days_elapsed_since_reg = cutoff
    
    return features


def features_to_dict(student_id: str, features: StudentFeatures) -> Dict:
    """Chuyển features thành dict để gửi đến API"""
    return {
        "student_id": student_id,
        "days_elapsed_since_reg": features.days_elapsed_since_reg,
        "clicks_per_day_total": features.clicks_per_day_total,
        "active_ratio_total": features.active_ratio_total,
        "avg_clicks_per_active_day_total": features.avg_clicks_per_active_day_total,
        "days_since_last_active": features.days_since_last_active,
        "clicks_last_14_days": features.clicks_last_14_days,
        "active_days_14": features.active_days_14,
        "clicks_per_day_14": features.clicks_per_day_14,
        "active_ratio_14": features.active_ratio_14,
        "clicks_last_7_days": features.clicks_last_7_days,
        "clicks_0_7": features.clicks_0_7,
        "clicks_8_14": features.clicks_8_14,
        "trend_click_14": features.trend_click_14,
        "ratio_click_14": features.ratio_click_14,
        "inactivity_streak_14": features.inactivity_streak_14,
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import requests
    
    # Giả sử BE lấy data từ database
    student_id = "12345"
    registration_date = datetime(2024, 1, 1)
    current_date = datetime(2024, 3, 1)
    days_since_reg = (current_date - registration_date).days
    
    # Giả sử đây là data từ DB
    activities = [
        VLEActivity(date=0, sum_click=100),
        VLEActivity(date=1, sum_click=80),
        VLEActivity(date=2, sum_click=90),
        VLEActivity(date=5, sum_click=120),
        VLEActivity(date=7, sum_click=70),
        VLEActivity(date=10, sum_click=60),
        VLEActivity(date=15, sum_click=50),
        VLEActivity(date=20, sum_click=40),
        VLEActivity(date=25, sum_click=30),
        VLEActivity(date=30, sum_click=20),
        VLEActivity(date=40, sum_click=10),
        VLEActivity(date=50, sum_click=5),
    ]
    
    # Tính features
    features = calculate_student_features(activities, days_since_reg)
    
    # Chuyển thành dict
    payload = features_to_dict(student_id, features)
    
    print("Features đã tính toán:")
    for key, value in payload.items():
        if key != "student_id":
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Gọi API
    print("\n=== Gọi API ===")
    response = requests.post(
        "http://localhost:8000/predict/features",
        json=[payload]  # Gửi list vì API nhận batch
    )
    
    if response.status_code == 200:
        result = response.json()[0]
        print(f"Student ID: {result['student_id']}")
        print(f"Dropout Probability: {result['dropout_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
    else:
        print(f"Error: {response.text}")
    
    print("\n=== Batch processing example ===")
    # Giả sử cronjob xử lý 1000 học viên
    batch_payload = []
    for i in range(1, 11):  # Demo với 10 học viên
        student_features = calculate_student_features(
            [VLEActivity(date=j, sum_click=50+i*5) for j in range(0, 60, 5)],
            days_since_reg=60
        )
        batch_payload.append(features_to_dict(f"STUDENT_{i:03d}", student_features))
    
    response = requests.post(
        "http://localhost:8000/predict/features",
        json=batch_payload
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"Processed {len(results)} students:")
        for result in results[:5]:  # Hiển thị 5 đầu tiên
            print(f"  {result['student_id']}: {result['dropout_probability']:.2%} - {result['risk_level']}")
    else:
        print(f"Error: {response.text}")


# ============================================================================
# JAVASCRIPT/TYPESCRIPT VERSION (Pseudo-code)
# ============================================================================

"""
// TypeScript implementation example

interface VLEActivity {
  date: number;
  sumClick: number;
}

interface StudentFeatures {
  student_id: string;
  days_elapsed_since_reg: number;
  clicks_per_day_total: number;
  active_ratio_total: number;
  avg_clicks_per_active_day_total: number;
  days_since_last_active: number;
  clicks_last_14_days: number;
  active_days_14: number;
  clicks_per_day_14: number;
  active_ratio_14: number;
  clicks_last_7_days: number;
  clicks_0_7: number;
  clicks_8_14: number;
  trend_click_14: number;
  ratio_click_14: number;
  inactivity_streak_14: number;
}

function computeInactivityStreak(
  activeDays: number[],
  startDay: number,
  endDay: number
): number {
  if (activeDays.length === 0) {
    return endDay - startDay + 1;
  }
  
  const activeSet = new Set(activeDays);
  let streak = 0;
  let day = endDay;
  
  while (day >= startDay && !activeSet.has(day)) {
    streak++;
    day--;
  }
  
  return streak;
}

function calculateStudentFeatures(
  activities: VLEActivity[],
  daysSinceRegistration: number,
  windowDays: number = 14,
  halfWindow: number = 7
): StudentFeatures {
  const cutoff = daysSinceRegistration;
  const wStart = Math.max(0, cutoff - (windowDays - 1));
  const wEnd = cutoff;
  
  // Filter activities
  const activitiesCum = activities.filter(a => a.date <= cutoff);
  const activitiesWin = activitiesCum.filter(a => a.date >= wStart);
  
  const features: Partial<StudentFeatures> = {
    days_elapsed_since_reg: cutoff,
  };
  
  // Cumulative features
  if (activitiesCum.length > 0) {
    const totalClicks = activitiesCum.reduce((sum, a) => sum + a.sumClick, 0);
    const activeDaysList = [...new Set(activitiesCum.map(a => a.date))];
    const activeDaysTotal = activeDaysList.length;
    const lastActive = Math.max(...activitiesCum.map(a => a.date));
    
    features.clicks_per_day_total = totalClicks / Math.max(cutoff, 1);
    features.active_ratio_total = activeDaysTotal / Math.max(cutoff, 1);
    features.days_since_last_active = cutoff - lastActive;
    features.avg_clicks_per_active_day_total = totalClicks / Math.max(activeDaysTotal, 1);
  } else {
    features.clicks_per_day_total = 0;
    features.active_ratio_total = 0;
    features.days_since_last_active = cutoff;
    features.avg_clicks_per_active_day_total = 0;
  }
  
  // Window features
  if (activitiesWin.length > 0) {
    const clicksLast14 = activitiesWin.reduce((sum, a) => sum + a.sumClick, 0);
    const activeDays14List = [...new Set(activitiesWin.map(a => a.date))];
    const activeDays14 = activeDays14List.length;
    
    features.clicks_last_14_days = clicksLast14;
    features.active_days_14 = activeDays14;
    features.clicks_per_day_14 = clicksLast14 / windowDays;
    features.active_ratio_14 = activeDays14 / windowDays;
    
    features.inactivity_streak_14 = computeInactivityStreak(
      activeDays14List.sort((a, b) => a - b),
      wStart,
      wEnd
    );
    
    // Split window
    const firstEnd = Math.min(wEnd, wStart + (halfWindow - 1));
    const secondStart = Math.min(wEnd, firstEnd + 1);
    
    const clicks07 = activitiesWin
      .filter(a => a.date >= wStart && a.date <= firstEnd)
      .reduce((sum, a) => sum + a.sumClick, 0);
    
    const clicks814 = activitiesWin
      .filter(a => a.date >= secondStart && a.date <= wEnd)
      .reduce((sum, a) => sum + a.sumClick, 0);
    
    features.clicks_0_7 = clicks07;
    features.clicks_8_14 = clicks814;
    features.trend_click_14 = clicks814 - clicks07;
    features.ratio_click_14 = (clicks814 + 1) / (clicks07 + 1);
  } else {
    features.clicks_last_14_days = 0;
    features.active_days_14 = 0;
    features.clicks_per_day_14 = 0;
    features.active_ratio_14 = 0;
    features.clicks_0_7 = 0;
    features.clicks_8_14 = 0;
    features.trend_click_14 = 0;
    features.ratio_click_14 = 1;
    features.inactivity_streak_14 = windowDays;
  }
  
  // Last 7 days
  if (activitiesCum.length > 0) {
    features.clicks_last_7_days = activitiesCum
      .filter(a => a.date > cutoff - 7)
      .reduce((sum, a) => sum + a.sumClick, 0);
  } else {
    features.clicks_last_7_days = 0;
  }
  
  return features as StudentFeatures;
}

// Usage in cronjob
async function runDropoutPredictionCronjob() {
  const students = await db.getActiveStudents();
  const batchSize = 100;
  
  for (let i = 0; i < students.length; i += batchSize) {
    const batch = students.slice(i, i + batchSize);
    const featuresPayload = [];
    
    for (const student of batch) {
      const activities = await db.getVLEActivities(student.id);
      const daysSinceReg = calculateDaysSinceRegistration(student.registrationDate);
      const features = calculateStudentFeatures(activities, daysSinceReg);
      
      featuresPayload.push({
        student_id: student.id,
        ...features
      });
    }
    
    // Call API
    const response = await fetch('http://ai-api:8000/predict/features', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(featuresPayload)
    });
    
    const predictions = await response.json();
    
    // Save to database
    await db.saveDropoutPredictions(predictions);
  }
}
"""


# ============================================================================
# JAVA VERSION (Pseudo-code)
# ============================================================================

"""
// Java implementation example

public class VLEActivity {
    private int date;
    private int sumClick;
    
    // constructors, getters, setters
}

public class StudentFeatures {
    private String studentId;
    private double daysElapsedSinceReg;
    private double clicksPerDayTotal;
    private double activeRatioTotal;
    private double avgClicksPerActiveDayTotal;
    private double daysSinceLastActive;
    private double clicksLast14Days;
    private double activeDays14;
    private double clicksPerDay14;
    private double activeRatio14;
    private double clicksLast7Days;
    private double clicks07;
    private double clicks814;
    private double trendClick14;
    private double ratioClick14;
    private double inactivityStreak14;
    
    // constructors, getters, setters
}

public class FeatureCalculator {
    
    private static final int WINDOW_DAYS = 14;
    private static final int HALF_WINDOW = 7;
    
    public static int computeInactivityStreak(List<Integer> activeDays, int startDay, int endDay) {
        if (activeDays.isEmpty()) {
            return endDay - startDay + 1;
        }
        
        Set<Integer> activeSet = new HashSet<>(activeDays);
        int streak = 0;
        int day = endDay;
        
        while (day >= startDay && !activeSet.contains(day)) {
            streak++;
            day--;
        }
        
        return streak;
    }
    
    public static StudentFeatures calculateStudentFeatures(
            String studentId,
            List<VLEActivity> activities,
            int daysSinceRegistration) {
        
        StudentFeatures features = new StudentFeatures();
        features.setStudentId(studentId);
        
        int cutoff = daysSinceRegistration;
        int wStart = Math.max(0, cutoff - (WINDOW_DAYS - 1));
        int wEnd = cutoff;
        
        // Filter activities
        List<VLEActivity> activitiesCum = activities.stream()
            .filter(a -> a.getDate() <= cutoff)
            .collect(Collectors.toList());
            
        List<VLEActivity> activitiesWin = activitiesCum.stream()
            .filter(a -> a.getDate() >= wStart)
            .collect(Collectors.toList());
        
        features.setDaysElapsedSinceReg(cutoff);
        
        // Cumulative features
        if (!activitiesCum.isEmpty()) {
            int totalClicks = activitiesCum.stream()
                .mapToInt(VLEActivity::getSumClick)
                .sum();
            
            long activeDaysTotal = activitiesCum.stream()
                .map(VLEActivity::getDate)
                .distinct()
                .count();
            
            int lastActive = activitiesCum.stream()
                .mapToInt(VLEActivity::getDate)
                .max()
                .getAsInt();
            
            features.setClicksPerDayTotal((double) totalClicks / Math.max(cutoff, 1));
            features.setActiveRatioTotal((double) activeDaysTotal / Math.max(cutoff, 1));
            features.setDaysSinceLastActive(cutoff - lastActive);
            features.setAvgClicksPerActiveDayTotal((double) totalClicks / Math.max(activeDaysTotal, 1));
        } else {
            features.setDaysSinceLastActive(cutoff);
        }
        
        // Window features
        if (!activitiesWin.isEmpty()) {
            int clicksLast14 = activitiesWin.stream()
                .mapToInt(VLEActivity::getSumClick)
                .sum();
            
            List<Integer> activeDays14List = activitiesWin.stream()
                .map(VLEActivity::getDate)
                .distinct()
                .collect(Collectors.toList());
            
            int activeDays14 = activeDays14List.size();
            
            features.setClicksLast14Days(clicksLast14);
            features.setActiveDays14(activeDays14);
            features.setClicksPerDay14((double) clicksLast14 / WINDOW_DAYS);
            features.setActiveRatio14((double) activeDays14 / WINDOW_DAYS);
            
            features.setInactivityStreak14(
                computeInactivityStreak(activeDays14List, wStart, wEnd)
            );
            
            // Split window logic...
            // (similar to Python version)
        } else {
            features.setInactivityStreak14(WINDOW_DAYS);
            features.setRatioClick14(1.0);
        }
        
        return features;
    }
}

// Usage in scheduled job
@Scheduled(cron = "0 0 2 * * *") // Run at 2 AM daily
public void runDropoutPredictionJob() {
    List<Student> students = studentRepository.findAllActive();
    int batchSize = 100;
    
    for (int i = 0; i < students.size(); i += batchSize) {
        List<Student> batch = students.subList(i, Math.min(i + batchSize, students.size()));
        List<StudentFeatures> featuresPayload = new ArrayList<>();
        
        for (Student student : batch) {
            List<VLEActivity> activities = vleRepository.findByStudentId(student.getId());
            int daysSinceReg = calculateDaysSinceRegistration(student.getRegistrationDate());
            
            StudentFeatures features = FeatureCalculator.calculateStudentFeatures(
                student.getId(),
                activities,
                daysSinceReg
            );
            
            featuresPayload.add(features);
        }
        
        // Call AI API
        List<DropoutPrediction> predictions = restTemplate.postForObject(
            "http://ai-api:8000/predict/features",
            featuresPayload,
            new ParameterizedTypeReference<List<DropoutPrediction>>() {}
        );
        
        // Save to database
        predictionRepository.saveAll(predictions);
    }
}
"""
