# Student Behavioral Disruption Prediction Model

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Technical Architecture](#technical-architecture)
- [Modeling Pipeline](#modeling-pipeline)
- [Key Findings](#key-findings)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)

---

## Project Overview

This project focuses on analyzing and predicting student behavioral disruptions within a school district to minimize class interruptions, improve educational outcomes, and support proactive disciplinary actions. Our primary stakeholder is a school board representative, seeking data-driven insights to identify at‑risk students and deploy targeted interventions.

### Core Objectives

- **Early Warning System**: Predict which students are likely to exhibit behavioral disruptions (referrals) in the coming week
- **Anomaly Detection**: Flag unusual spikes in referrals for individual students or classrooms
- **Interpretability**: Provide clear explanations of why certain students are at higher risk
- **Data Enrichment**: Incorporate external sources (weather, family engagement surveys) to boost model accuracy

---

## Problem Statement

Educational institutions face significant challenges in managing student behavior effectively. Traditional reactive approaches often result in:

- **Delayed Interventions**: Late identification of at-risk students
- **Resource Misallocation**: Inefficient use of support resources
- **Academic Impact**: Disruptions affecting learning outcomes
- **Administrative Burden**: Manual monitoring and intervention processes

This project addresses these challenges through:

- **Predictive Analytics**: Proactive identification of behavioral risks
- **Data-Driven Insights**: Evidence-based intervention strategies
- **Automated Monitoring**: Scalable surveillance of student behavior patterns
- **Intervention Optimization**: Targeted resource allocation for maximum impact

---

## Dataset Overview

We integrate four key data sources at a **student‑week** level to create a comprehensive behavioral prediction system:

### 1. Bus Conduct Data

- **Incident Records**: Detailed reports of behavioral incidents on school buses
- **Incident Types**: Categorization of misconduct (disruption, fighting, vandalism, etc.)
- **Route Information**: Bus route, driver, and timing data
- **Student Demographics**: Grade level, gender, ethnicity, socioeconomic indicators

### 2. Disciplinary Referral Data

- **Staff Reports**: Teacher and administrator referral submissions
- **Severity Classification**: Minor, moderate, and major behavioral violations
- **Behavior Categories**: Academic dishonesty, disruption, fighting, substance use, etc.
- **Temporal Data**: Time of day, day of week, and seasonal patterns
- **Location Tracking**: Classroom, hallway, cafeteria, or other school areas

### 3. Family Engagement Data

- **Survey Responses**: Parental involvement and satisfaction metrics
- **Resource Access**: Family access to educational and support resources
- **Communication Patterns**: Frequency and quality of school-family interactions
- **Support Needs**: Identified family support requirements

### 4. Weather Data

- **Daily Metrics**: Temperature, humidity, precipitation, wind speed
- **Severe Weather Indicators**: Storm warnings, extreme temperature flags
- **Weekly Aggregation**: Averaged weather conditions by week
- **Seasonal Patterns**: Weather impact on student behavior

### Data Integration Strategy

- **Student Identifier**: Primary key for merging all data sources
- **Temporal Alignment**: Weekly aggregation for consistent time periods
- **Missing Value Handling**: Strategic imputation for incomplete records
- **Feature Standardization**: Consistent encoding across all sources

---

## Technical Architecture

### Data Pipeline

```
Raw Data Sources → Data Cleaning → Feature Engineering → Model Training → Prediction Output
```

### System Components

- **Data Ingestion**: Automated collection from multiple school systems
- **Preprocessing Engine**: Standardized cleaning and feature creation
- **Model Training Pipeline**: Automated model selection and optimization
- **Prediction Service**: Real-time risk assessment and alerting
- **Visualization Dashboard**: Interactive reporting and trend analysis

### Technology Stack

| Component                | Technology          | Purpose                              |
| ------------------------ | ------------------- | ------------------------------------ |
| **Data Processing**      | Pandas, NumPy       | Data manipulation and analysis       |
| **Machine Learning**     | Scikit-learn        | Model training and evaluation        |
| **Visualization**        | Matplotlib, Seaborn | Data visualization and reporting     |
| **Imbalanced Learning**  | Imbalanced-learn    | Handling class imbalance             |
| **Statistical Analysis** | SciPy, Statsmodels  | Statistical testing and validation   |
| **Development**          | Jupyter Notebook    | Interactive development and analysis |

---

## Modeling Pipeline

### 1. Data Integration & Cleaning

**Data Merging Strategy**:

- Merge all sources on `Student Identifier` and `Week`
- Handle missing values with domain-appropriate strategies
- Standardize categorical entries across all data sources
- Validate data quality and consistency

**Data Quality Assurance**:

- Duplicate detection and removal
- Outlier identification and treatment
- Data type validation and conversion
- Cross-source consistency checks

### 2. Feature Engineering

#### Behavioral Features

- **`weekly_referrals`**: Current-week referral count per student
- **`bus_incident_counts`**: Transportation-related behavioral incidents
- **`referral_severity_avg`**: Average severity of recent referrals
- **`behavior_type_frequency`**: Frequency of specific behavior types

#### Demographic Features

- **One-hot encoding**: Grade level, gender, ethnicity categories
- **Socioeconomic indicators**: Free/reduced lunch status
- **Academic indicators**: GPA trends, attendance patterns
- **Family structure**: Household composition indicators

#### External Enrichment Features

- **Weather metrics**: Weekly average temperature, peak humidity
- **Severe weather flags**: Storm warnings, extreme conditions
- **Seasonal patterns**: Academic calendar events, holidays
- **Environmental factors**: Air quality, temperature extremes

#### Temporal Features

- **Lag features**: Previous week's behavioral indicators
- **Rolling averages**: 2-week, 4-week behavioral trends
- **Day-of-week patterns**: Behavioral patterns by weekday
- **Seasonal trends**: Academic year progression effects

### 3. Target Definition

#### Classification Target

- **`referral_next_week`**: Binary indicator (0/1) for any referral in the following week
- **Primary focus**: Early warning system for behavioral risk

#### Regression Target

- **`referrals_next_week_count`**: Numerical prediction of referral count
- **Secondary focus**: Quantifying expected behavioral severity

### 4. Preprocessing Strategy

#### Feature Transformation

- **ColumnTransformer**: Automated scaling and encoding pipeline
- **Numeric scaling**: StandardScaler for continuous features
- **Categorical encoding**: OneHotEncoder for categorical variables
- **Feature selection**: Correlation analysis and importance ranking

#### Class Balancing

- **SMOTE**: Synthetic Minority Over-sampling Technique
- **Balanced sampling**: Ensures adequate representation of positive cases
- **Cross-validation**: Stratified sampling for robust evaluation

### 5. Model Selection & Training

#### Baseline Models

1. **Logistic Regression**

   - Linear model for binary risk classification
   - Interpretable coefficients and feature importance
   - Fast training and prediction
   - Good baseline for comparison

2. **Random Forest**

   - Ensemble of decision trees capturing non-linear patterns
   - Feature importance ranking
   - Robust to outliers and noise
   - Handles mixed data types well

3. **MLP Neural Network**
   - Multi-layer perceptron for complex feature interactions
   - Non-linear pattern recognition
   - High capacity for learning complex relationships
   - Our top-performing model

#### Hyperparameter Optimization

- **GridSearchCV**: Systematic parameter search
- **Optimization metrics**: F1-score and ROC-AUC focus
- **Cross-validation**: 5-fold CV for robust evaluation
- **Early stopping**: Prevent overfitting in neural networks

### 6. Evaluation Framework

#### Classification Metrics

- **Precision**: Positive predictive value for risk identification
- **Recall**: True positive rate for capturing actual incidents
- **F1-score**: Harmonic mean balancing precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Confusion matrices**: Detailed error analysis

#### Regression Metrics

- **RMSE**: Root Mean Square Error for count predictions
- **R²**: Coefficient of determination for model fit
- **MAE**: Mean Absolute Error for practical interpretation

---

## Key Findings

### Model Performance Summary

| Model                   | F1-Score | ROC-AUC | Precision | Recall | Training Time |
| ----------------------- | -------- | ------- | --------- | ------ | ------------- |
| **MLP Neural Network**  | 0.82     | 0.89    | 0.79      | 0.85   | 45s           |
| **Random Forest**       | 0.80     | 0.87    | 0.77      | 0.83   | 12s           |
| **Logistic Regression** | 0.76     | 0.84    | 0.74      | 0.78   | 3s            |

### Critical Insights

1. **MLP Superiority**: Neural network achieved highest classification performance by capturing complex non-linear interactions among behavioral, demographic, and environmental features

2. **Feature Importance**:

   - **Previous week referrals** (most predictive)
   - **Bus incident count** (strong indicator)
   - **Average temperature** (moderate impact)
   - **Family engagement levels** (contextual factor)

3. **Temporal Patterns**:

   - Strong correlation between current and future week behavior
   - Seasonal effects on behavioral patterns
   - Day-of-week variations in incident frequency

4. **Intervention Opportunities**:
   - Early identification of at-risk students (1-week advance warning)
   - Targeted resource allocation based on risk factors
   - Preventive measures for high-risk time periods

### Practical Applications

- **Weekly Risk Reports**: Automated generation of student risk assessments
- **Intervention Prioritization**: Ranking students by intervention urgency
- **Resource Planning**: Data-driven allocation of support resources
- **Trend Analysis**: Long-term behavioral pattern identification

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- 8GB+ RAM (for neural network training)
- Internet connection for package installation

### Environment Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd student-referral-predictor
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import pandas, sklearn, matplotlib; print('Setup complete!')"
   ```

---

## Usage Instructions

### Quick Start

1. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. **Open the main pipeline**: `Behavioral_Disruptions_Pipeline.ipynb`

3. **Run all cells**: Execute the complete analysis pipeline

### Step-by-Step Process

1. **Data Loading and Preprocessing**:

   - Load combined datasets from CSV files
   - Apply data cleaning and standardization
   - Create engineered features

2. **Model Training**:

   - Execute SMOTE balancing for class imbalance
   - Train multiple models with cross-validation
   - Perform hyperparameter optimization

3. **Results Analysis**:
   - Review printed performance metrics
   - Examine ROC curves and confusion matrices
   - Analyze feature importance rankings

### Customization Options

- **Feature Engineering**: Modify feature creation in the preprocessing section
- **Model Selection**: Add or remove models from the pipeline
- **Evaluation Metrics**: Customize performance assessment criteria
- **Visualization**: Adjust plots and charts for specific reporting needs

---

## Model Performance

### Classification Results

The MLP neural network achieved the best overall performance:

- **F1-Score**: 0.82 (excellent balance of precision and recall)
- **ROC-AUC**: 0.89 (strong discriminative ability)
- **Precision**: 0.79 (high positive predictive value)
- **Recall**: 0.85 (excellent sensitivity for risk detection)

### Feature Importance Analysis

Top predictive features (Random Forest importance):

1. **Previous week referrals** (0.28)
2. **Bus incident count** (0.22)
3. **Average temperature** (0.15)
4. **Family engagement score** (0.12)
5. **Student grade level** (0.08)

### Model Interpretability

- **Logistic Regression**: Clear coefficient interpretation
- **Random Forest**: Feature importance rankings
- **MLP**: Black-box model requiring additional analysis
- **SHAP Values**: Model-agnostic interpretability (future enhancement)

---

## Future Enhancements

### Model Improvements

- **Deep Learning**: Advanced neural network architectures
- **Ensemble Methods**: Combining multiple model predictions
- **Time Series**: LSTM/GRU for temporal pattern recognition
- **Transfer Learning**: Pre-trained models for behavioral prediction

### Deployment Features

- **Real-time API**: RESTful service for live predictions
- **Automated Retraining**: Scheduled model updates with new data
- **A/B Testing**: Framework for model comparison in production
- **Monitoring Dashboard**: Real-time model performance tracking

### Data Enhancements

- **Additional Sources**: Academic performance, attendance records
- **External Data**: Community factors, economic indicators
- **Real-time Feeds**: Live weather and event data
- **Multi-modal Data**: Text analysis of incident reports

### Advanced Analytics

- **Causal Inference**: Understanding intervention effectiveness
- **Prescriptive Analytics**: Recommended intervention strategies
- **Risk Stratification**: Multi-level risk categorization
- **Predictive Maintenance**: Model performance monitoring

---

## GitHub Actions Auto-Clean Feature

> _Maintaining clean notebooks on each pull request._

### Automated Workflow

1. **Code Checkout**: Retrieves latest code from PR branch
2. **Notebook Cleaning**: Strips output and metadata from modified notebooks
3. **Clean Commit**: Commits cleaned notebooks back to the PR branch
4. **Quality Assurance**: Ensures consistent notebook state across contributors

### Benefits

- **Version Control**: Clean notebook history in repository
- **Collaboration**: Consistent notebook state for team members
- **Automation**: No manual cleaning required
- **Quality**: Professional repository presentation

---

_This project demonstrates advanced machine learning techniques applied to real-world educational challenges, showcasing skills in predictive modeling, data engineering, and practical AI applications._
