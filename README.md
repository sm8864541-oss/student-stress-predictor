# 🧠 Student Stress Level Predictor with Explainable AI

A machine learning mini-project that predicts a student's stress level
(Low / Medium / High) from lifestyle habits — and, unlike most
prediction projects, **explains why** each prediction was made using
SHAP (SHapley Additive exPlanations).

## Why this project is different

Most stress/performance prediction projects stop at "here's the
accuracy." This project adds an **interpretability layer**: for every
prediction, you can see exactly which factors (sleep, screen time,
academic pressure, etc.) pushed the score up or down, and by how much.
This makes the model's decisions transparent and trustworthy —
something increasingly important in ML systems that affect real
people.

## Features

- Random Forest classifier trained on lifestyle/habit data
- Global explanation: which factors matter most across all students
- Local explanation: why *this specific student* got *this specific*
  prediction (SHAP waterfall plot)
- Interactive Streamlit dashboard — enter your own habits, get an
  instant prediction + explanation

## Tech Stack

Python · pandas · scikit-learn · SHAP · Streamlit · matplotlib

## Project Structure

```
student-stress-predictor/
├── data/
│   └── student_stress.csv       # dataset (sample or real Kaggle data)
├── model/
│   ├── rf_model.pkl              # trained model (generated)
│   ├── label_encoder.pkl         # generated
│   └── feature_names.pkl         # generated
├── generate_sample_data.py       # creates synthetic test data
├── train_model.py                # trains and saves the model
├── explain_model.py              # generates SHAP explanation plots
├── app.py                        # Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

## Setup & Usage

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd student-stress-predictor
   pip install -r requirements.txt
   ```

2. **Get data** — two options:
   - **Quick test:** run the included generator to create a synthetic
     dataset instantly:
     ```bash
     python generate_sample_data.py
     ```
   - **Real data:** download the [Student Stress Factors dataset](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis)
     from Kaggle, and save it as `data/student_stress.csv` (rename
     columns to match the ones used in `train_model.py` if needed).

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Generate SHAP explanations**
   ```bash
   python explain_model.py
   ```
   Check `model/global_importance.png` and
   `model/individual_explanation.png`.

5. **Run the interactive dashboard**
   ```bash
   streamlit run app.py
   ```

## Results

With the Random Forest model on the sample dataset:
- Overall accuracy: ~67-70% across 3 classes (Low/Medium/High)
- SHAP analysis shows sleep hours and academic pressure are typically
  the strongest predictors of high stress

*(Update this section with your real dataset's results once you run
it.)*

## Future Improvements

- Compare multiple models (Logistic Regression, XGBoost) for accuracy
- Add a larger, real-world survey dataset
- Deploy the Streamlit app publicly (Streamlit Community Cloud)

## Author

[Your Name] — BTech IT, 2nd Year

## License

MIT
