# Data Mining Project: Predicting and Minimizing Student Behavioral Disruptions

## Introduction

This project focuses on analyzing and predicting student behavioral disruptions within a school district to minimize class interruptions, improve educational outcomes, and support proactive disciplinary actions. Our primary stakeholder is a school board representative, seeking data-driven insights to identify at‑risk students and deploy targeted interventions.

## Project Goals

* **Early Warning**: Predict which students are likely to exhibit behavioral disruptions (referrals) in the coming week.
* **Anomaly Detection**: Flag unusual spikes in referrals for individual students or classrooms.
* **Interpretability**: Provide clear explanations of why certain students are at higher risk.
* **Data Enrichment**: Incorporate external sources (weather, family engagement surveys) to boost model accuracy.

## Dataset Overview

We integrate four key data sources at a **student‑week** level:

1. **Bus Conduct Data**

   * Incident records on school buses, including type of incident, bus route, and driver response.
   * Student demographics: grade level, gender, ethnicity.

2. **Disciplinary Referral Data**

   * Staff‑reported referrals, categorized by severity and behavior type.
   * Time of day and location of each incident.

3. **Family Engagement Data**

   * Survey responses capturing parental involvement, satisfaction, and resource access.

4. **Weather Data**

   * Daily temperature, humidity, and severe weather indicators aggregated by week.

## Modeling Pipeline

1. **Data Integration & Cleaning**

   * Merge all sources on `Student Identifier` and `Week`.
   * Handle missing values and standardize categorical entries.

2. **Feature Engineering**

   * **Lag Features**: `weekly_referrals` (current-week count), `bus_incident_counts`.
   * **Demographics**: One‑hot encoding for grade, gender, ethnicity, free/reduced lunch status.
   * **External Enrichment**: Weekly average temperature, peak humidity, severe weather flags.

3. **Target Definition**

   * **Classification**: `referral_next_week` (0/1) indicates whether a student receives any referral in the following week.
   * **Regression**: `referrals_next_week_count` for forecasting the number of referrals this week.

4. **Preprocessing**

   * Use a `ColumnTransformer` to scale numeric features and one‑hot encode categoricals.
   * Employ SMOTE to balance the minority class in the classification pipeline.

5. **Model Selection & Training**

   * **Logistic Regression**: Baseline linear model for binary risk classification.
   * **Random Forest**: Ensemble of decision trees capturing non‑linear splits.
   * **MLP Neural Network**: Multi‑layer perceptron to learn complex feature interactions (our top performer).
   * Hyperparameter tuning via `GridSearchCV` on F1-score and ROC‑AUC.

6. **Evaluation**

   * **Classification Metrics**: Precision, recall, F1-score, ROC‑AUC, and confusion matrices.
   * **Regression Metrics**: RMSE and R² (for numerical forecasts).

## How It Works

1. **Launch Notebook**: Open `Behavioral_Disruptions_Pipeline.ipynb`.
2. **Preprocessing**: Run cells to load combined datasets, apply cleaning, and create features.
3. **Train Models**: Execute the classification pipeline—SMOTE balancing, model training, and grid search.
4. **Review Results**: Check printed metrics, plots of ROC curves, and feature importances.

## Usage

```bash
# Clone repository
git clone git@github.com:TDRobertson/behavioral-disruptions-predictor.git
cd behavioral-disruptions-predictor

# Set up environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the pipeline by hitting run all cells in the Jupyter notebook
```

## Key Findings & Next Steps

* The **MLP neural network** achieved the highest classification performance (F1 and ROC‑AUC), capturing non‑linear interactions among behavioral, demographic, and environmental features.
* **Random Forests** provided strong interpretability via feature importances and performed close behind the neural network.
* Future work includes deploying the trained classifier to a streaming pipeline and extending the regression module to forecast exact referral counts.

## GitHub Actions Auto-Clean Feature

> *Maintaining clean notebooks on each pull request.*

1. Checks out code and installs `nbstripout`.
2. Strips output and metadata from modified notebooks.
3. Commits cleaned notebooks back to the PR branch.

---