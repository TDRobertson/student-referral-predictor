# Student Behavioral Disruption Prediction Model

[![CI - Strip Notebook Outputs](https://github.com/TDRobertson/student-referral-predictor/actions/workflows/strip-notebook-output.yml/badge.svg)](https://github.com/TDRobertson/student-referral-predictor/actions/workflows/strip-notebook-output.yml)

> Predicting which students are likely to receive a behavioral referral in the coming week — one week before it happens.

## About the Project

School districts respond to behavioral disruptions after they occur. A referral gets filed, a student gets sent to the office, and an intervention follows the incident. This project was built around a different question: what if you could identify which students were heading toward a disruption before it happened?

Working with real anonymized records from an actual Tennessee school district, this pipeline predicts behavioral referrals at the **student-week** level — flagging at-risk students seven days in advance. The primary stakeholder was a school board representative looking for data-driven tools to move from reactive discipline to proactive, targeted intervention.

The pipeline integrates four data sources — disciplinary referrals, bus conduct reports, family engagement surveys, and daily weather data — into a unified student-week feature set. Before any model was trained, six statistical hypotheses were tested to validate which external factors genuinely correlate with behavioral patterns. The result is a prediction system grounded in evidence, not assumption.

The top-performing model — a multi-layer perceptron neural network — achieved an F1-score of **0.82** and a ROC-AUC of **0.89** on a held-out test set. The full pipeline is published at [github.com/TDRobertson/student-referral-predictor](https://github.com/TDRobertson/student-referral-predictor).

---

## Table of Contents

- [The Problem](#the-problem)
- [Dataset Overview](#dataset-overview)
- [Technical Architecture](#technical-architecture)
- [Statistical Hypothesis Testing](#statistical-hypothesis-testing)
- [Modeling Pipeline](#modeling-pipeline)
- [Results](#results)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [CI/CD — Automated Notebook Cleaning](#cicd--automated-notebook-cleaning)
- [Acknowledgments](#acknowledgments)

---

## The Problem

Educational institutions have a structural challenge: the tools available to identify struggling students are almost entirely reactive. Referrals, suspensions, and disciplinary records are generated *after* a behavioral episode — not before it. Interventions arrive late, support resources are allocated in response to crises rather than in anticipation of them, and the students most in need of help are often identified only after a pattern has already set in.

This project addresses that gap directly:

- **Delayed Interventions**: At-risk students are identified after incidents occur, not before
- **Resource Misallocation**: Counselor and administrative time is deployed reactively rather than strategically
- **Academic Impact**: Behavioral disruptions affect classroom learning for the student, peers, and instructors
- **No Predictive Signal Utilized**: Behavioral patterns, transportation incidents, and family engagement data exist but aren't connected into a predictive system

---

## Dataset Overview

The pipeline integrates four real data sources, all provided as anonymized records from an actual school district. Data is aggregated to the **student-week** level — the unit at which the prediction is made — producing a final model-ready dataset of **860 student-week observations**.

### 1. Disciplinary Referral Data — 1,956 records

Staff-submitted referrals covering the full academic year. Each record includes the student identifier, behavior type (academic dishonesty, disruption, fighting, substance use, etc.), severity classification (Minor / Moderate / Major), location, time of day, and day of week. This is the primary behavioral signal and the source of the prediction target (`referral_next_week`).

### 2. Bus Conduct Data — 169 records

Behavioral incidents reported by bus drivers and transportation staff. Records include incident type, bus route, timing, and student demographics. Bus conduct turns out to be a stronger predictor than its record volume suggests — students who generate transportation incidents show distinct behavioral patterns from those whose incidents are confined to the classroom.

### 3. Family Engagement Survey Data — 258 responses

Parent and guardian survey responses covering school-family communication, resource access, academic support, and school environment satisfaction. The survey spans 40+ questions. Family engagement level becomes a contextual feature — one of the six hypotheses tested was whether higher engagement correlates with fewer referrals. It does.

### 4. Weather Data — 427 days

Daily weather records including temperature (max/min), humidity, precipitation, wind speed, and sea level pressure. Weather data is aggregated to the weekly level to align with the student-week prediction frame. Its inclusion wasn't assumed — it was validated through hypothesis testing before being incorporated into the feature set.

### Data Integration

- **Primary key**: Anonymized student identifier (consistent across referral, bus, and survey datasets)
- **Temporal alignment**: All sources aggregated to weekly boundaries
- **Target definition**: Features reflect the current week; the prediction target is the following week
- **Missing values**: Domain-appropriate imputation — no zero-fills for behavioral counts

---

## Technical Architecture

The pipeline runs as a single Jupyter notebook (`Behavioral_Disruptions_Pipeline.ipynb`) with clearly separated stages. Each stage produces a clean output that feeds the next, making the pipeline auditable and reproducible end-to-end.

```
Raw Data Sources → Cleaning & Integration → Feature Engineering → Hypothesis Testing → Model Training → Evaluation
```

### Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Data Processing** | Pandas ≥1.5.0, NumPy ≥1.24.0 | Integration, cleaning, aggregation |
| **Machine Learning** | Scikit-learn ≥1.2.0 | Preprocessing pipelines, model training, evaluation |
| **Class Balancing** | Imbalanced-learn ≥0.11.0 | SMOTE for minority class oversampling |
| **Statistical Analysis** | SciPy ≥1.10.0, Statsmodels ≥0.13.0 | Hypothesis testing (ANOVA, t-tests, correlations) |
| **Post-Hoc Testing** | Scikit-posthocs ≥0.11.4 | Dunn's test for multi-group comparisons |
| **Visualization** | Matplotlib ≥3.7.0, Seaborn ≥0.12.0 | EDA plots, model evaluation visualizations |
| **Development** | Jupyter Notebook ≥7.0.0 | Interactive pipeline and analysis |

---

## Statistical Hypothesis Testing

Before any model was trained, six behavioral hypotheses were tested against the data. This was a deliberate design decision: if weather or family engagement don't actually correlate with behavioral patterns in this district's records, they don't belong in the feature set. The hypothesis testing phase confirmed which external factors were real signals — not assumptions.

| # | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | Referral frequency increases near testing season | One-way ANOVA | **Significant** — F = 3.46, p = 0.0004; strong academic calendar effect confirmed |
| H2 | Bus misconduct associated with more in-class referrals | Welch's t-test | Counterintuitive — bus-incident students had *fewer* classroom referrals (possible classroom removal effect) |
| H3 | Family engagement negatively correlates with referrals | Spearman correlation | **Confirmed** — higher engagement associated with fewer referrals |
| H4 | Total referral count differs by grade level | One-way ANOVA | **Confirmed** — significant grade-level variation in referral frequency |
| H5 | Referral volume correlates with weather factors | Pearson correlation | **Confirmed** — temperature and humidity show measurable behavioral correlation |
| H6 | Referral volume correlates with temperature extremes and school breaks | Dunn's post-hoc test | **Confirmed** — elevated referrals around transitions and weather extremes |

The counterintuitive finding from H2 — that bus-incident students had *lower* in-class referral rates — shaped how transportation features were interpreted in the final model. Students removed from classroom environments for behavioral reasons may simply have fewer opportunities to generate classroom referrals, not fewer behavioral incidents overall.

---

## Modeling Pipeline

### Feature Engineering

All four data sources were aggregated to the student-week level before modeling. The final feature set covers three categories:

**Behavioral features**: `weekly_referrals`, `bus_incident_count`, `referral_severity_avg`, behavior type frequency distributions

**Demographic features**: Grade level, gender, ethnicity, free/reduced lunch status — one-hot encoded

**External enrichment features**: Weekly average temperature, humidity, precipitation, wind gust, sea level pressure; family engagement survey aggregates

### Preprocessing Strategy

- **ColumnTransformer**: StandardScaler for numeric features; OneHotEncoder for categorical features
- **SMOTE**: Applied to the training set only — never the full dataset — to prevent synthetic test samples from inflating evaluation metrics
- **Train/test split**: 80/20 stratified split to preserve the class ratio across both sets

### Models

Three classifiers were trained and compared, spanning the interpretability-to-performance spectrum:

| Model | Characteristics |
|---|---|
| **Logistic Regression** | Baseline linear classifier; interpretable coefficients; establishes the performance floor |
| **Random Forest** | Ensemble of decision trees; captures non-linear patterns; provides feature importance ranking |
| **MLP Neural Network** | Multi-layer perceptron; highest capacity for complex feature interactions; best overall performance |

All models were tuned using GridSearchCV with 5-fold stratified cross-validation, optimizing jointly for F1-score and ROC-AUC.

### Anomaly Detection

DBSCAN clustering was applied to identify unusual referral spikes — students or time periods with behavioral patterns that fall outside the normal distribution. This surfaces cases that may require immediate administrative attention independent of the weekly prediction cycle.

---

## Results

### Model Performance

| Model | F1-Score | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| **MLP Neural Network** | **0.82** | **0.89** | 0.79 | 0.85 |
| Random Forest | 0.80 | 0.87 | 0.77 | 0.83 |
| Logistic Regression | 0.76 | 0.84 | 0.74 | 0.78 |

The MLP outperformed the other models by capturing complex non-linear interactions among behavioral, demographic, and environmental features that a linear model can't represent and a forest approximates less precisely. An ROC-AUC of 0.89 indicates strong discriminative ability — the model reliably distinguishes at-risk students from those unlikely to receive a referral the following week.

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Previous week referrals | 0.28 |
| Bus incident count | 0.22 |
| Average temperature | 0.15 |
| Family engagement score | 0.12 |
| Student grade level | 0.08 |

The dominant signal is temporal persistence: students who had referrals this week are more likely to have referrals next week. This validates the core premise of an early warning system — behavioral patterns are not random, and they're detectable in advance with enough lead time to intervene.

---

## Installation and Setup

**Requirements**: Python 3.8+, Jupyter Notebook or JupyterLab, 8GB+ RAM

```bash
# Clone the repository
git clone https://github.com/TDRobertson/student-referral-predictor.git
cd student-referral-predictor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify the environment
python -c "import pandas, sklearn, matplotlib; print('Setup complete!')"
```

---

## Usage

```bash
jupyter notebook
```

Open `Behavioral_Disruptions_Pipeline.ipynb` and run all cells to execute the complete pipeline: data loading → cleaning → EDA → hypothesis testing → feature engineering → model training → evaluation.

The notebook is self-contained. All data files are included in the repository. Output cells are stripped on each commit (see below) — run the notebook locally to regenerate visualizations and model results.

---

## CI/CD — Automated Notebook Cleaning

The repository uses a GitHub Actions workflow (`.github/workflows/strip-notebook-output.yml`) that automatically strips output cells and execution metadata from any modified Jupyter notebooks on each pull request to `main`.

The workflow checks out the PR branch, runs `nbstripout` on all modified notebooks, and commits the cleaned versions back to the branch automatically. This keeps the repository history clean — no serialized outputs, no large binary diffs from notebook renders, no stale execution counts committed alongside code changes. Contributors don't need to manually clear outputs before submitting.

---

## Acknowledgments

Built for **CSC 4200 — Data Mining** at Tennessee Tech University, Spring 2024. School district data was provided by a real stakeholder for research purposes; all student identifiers are anonymized.

**Contributors**: Thomas Robertson, Claudia Nething
