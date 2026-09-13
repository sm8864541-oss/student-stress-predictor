"""
generate_sample_data.py
------------------------
Creates a synthetic student-stress dataset so you can run the whole
pipeline TODAY while you wait to download the real Kaggle dataset.

Once you have the real CSV (from Kaggle), just place it at
data/student_stress.csv with matching column names, and everything
else in this project works without changes.

Run:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 800  # number of fake students

sleep_hours = np.random.normal(6.5, 1.5, N).clip(2, 10)
study_hours = np.random.normal(4, 2, N).clip(0, 12)
screen_time = np.random.normal(6, 2.5, N).clip(0, 14)
social_activity = np.random.randint(0, 11, N)          # 0-10 scale
academic_pressure = np.random.randint(1, 6, N)          # 1-5 scale
extracurricular_hours = np.random.normal(2, 1.5, N).clip(0, 8)

# Create a "stress score" using a hidden formula (this is what the
# model will try to learn), then add noise so it's realistic.
raw_score = (
    (10 - sleep_hours) * 1.2
    + academic_pressure * 1.5
    + screen_time * 0.6
    - social_activity * 0.4
    - extracurricular_hours * 0.3
    + np.random.normal(0, 2, N)
)

# Convert to 3 classes: Low / Medium / High stress
stress_level = pd.qcut(raw_score, q=3, labels=["Low", "Medium", "High"])

df = pd.DataFrame({
    "sleep_hours": sleep_hours.round(1),
    "study_hours": study_hours.round(1),
    "screen_time": screen_time.round(1),
    "social_activity": social_activity,
    "academic_pressure": academic_pressure,
    "extracurricular_hours": extracurricular_hours.round(1),
    "stress_level": stress_level,
})

df.to_csv("data/student_stress.csv", index=False)
print(f"Sample dataset created: data/student_stress.csv ({len(df)} rows)")
print(df.head())
