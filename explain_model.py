"""
explain_model.py
-----------------
This is the "unique" part of the project. Instead of just showing
accuracy, we use SHAP (SHapley Additive exPlanations) to show WHICH
features drove each prediction and by how much. This is what you
write about in your IEEE-style report as your contribution.

Run:
    python explain_model.py

Produces:
    - A summary plot (global feature importance across all students)
    - A force/waterfall plot for one individual student's prediction
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load trained model + data
model = joblib.load("model/rf_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_names = joblib.load("model/feature_names.pkl")

df = pd.read_csv("data/student_stress.csv")
X = df[feature_names]

# 2. Create SHAP explainer for the Random Forest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 3. GLOBAL explanation: which features matter most overall?
# shap_values is a list (one array per class) for multi-class RF
class_index = list(label_encoder.classes_).index("High")  # focus on "High" stress class

plt.figure()
shap.summary_plot(shap_values[:, :, class_index], X, show=False)
plt.title("Which factors drive HIGH stress predictions overall?")
plt.tight_layout()
plt.savefig("model/global_importance.png", dpi=150)
print("Saved: model/global_importance.png")

# 4. LOCAL explanation: explain ONE student's prediction (row 0)
sample_idx = 0
plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx, :, class_index],
        base_values=explainer.expected_value[class_index],
        data=X.iloc[sample_idx],
        feature_names=feature_names,
    ),
    show=False,
)
plt.title(f"Why student #{sample_idx} got this prediction")
plt.tight_layout()
plt.savefig("model/individual_explanation.png", dpi=150)
print("Saved: model/individual_explanation.png")

print("\nOpen the two PNG files in model/ to see the explanations.")
