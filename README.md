# Data Mining Project: Predicting and Minimizing Student Behavioral Disruptions

## Introduction

This project focuses on analyzing and predicting student behavioral disruptions within a school district to minimize class disruptions, improve educational outcomes, and support anticipatory disciplinary actions. Our primary customer is represented by Adam West, aiming to proactively address behavioral issues using data-driven approaches.

## Project Goals

The overarching objective is to identify patterns, anomalies, and predictive indicators for student behavioral disruptions. Specific goals include:

- Predicting which students are likely to exhibit behavioral disruptions within the coming week.
- Identifying anomalous patterns involving specific students or teachers (over-referrals).
- Providing clear, interpretable insights into why certain behavioral issues occur.
- Augmenting existing datasets with external information (weather, socio-economic indicators) to enhance predictive accuracy.

## Dataset Overview

The project employs three primary datasets provided by the school district:

### 1. Bus Conduct Data
- Records incidents occurring on school buses.
- Fields include date, bus route type, type of incident, driver's actions, student demographics (ethnicity, gender, grade level), and identifiers.

### 2. Family Engagement Data
- Contains survey responses from parents/guardians regarding their engagement with the school system.
- Includes demographics, attitudes towards school recommendation, access to resources, preferences for engagement events, and satisfaction levels.

### 3. Disciplinary Referral Data
- Captures incidents reported by school staff, categorizing the severity and nature of incidents (minor/major).
- Includes time, location of incident, specific behaviors, and student demographics (ethnicity, gender, grade level, lunch status, and identifiers).

## Methodological Approach

### Phase 1: Exploratory Data Analysis (EDA) & Hypothesis Investigation (~1 week)
- **Data Cleaning:** Standardizing dates, addressing missing values, and correcting inconsistencies.
- **Descriptive Analysis:** Identifying patterns related to time (months, days, times), frequency, and location of referrals.
- **Hypothesis Generation & Testing:**
  - Investigate temporal patterns (time-based incidents).
  - Relationship between bus incidents and classroom referrals.
  - Impact of family engagement levels on student behavior.
- Apply statistical methods (Chi-square, t-tests, ANOVA) to confirm/refute hypotheses.

### Phase 2: Model Development & Follow-up Investigation (~1 week)
- **Feature Engineering:**
  - Student-level aggregation (frequency and severity of incidents).
  - Integration of external data sources (weather, neighborhood demographics).
- **Modeling Approaches:**
  - Logistic regression (classification of disruptive vs. non-disruptive students).
  - Linear regression (prediction of incident severity or frequency).
  - Advanced modeling (Decision Trees, Random Forests, Gradient Boosting, clustering for anomaly detection).
- **Model Evaluation:**
  - Precision, recall, F1-score, ROC-AUC for classification.
  - RMSE and R² metrics for regression models.
  - Clearly interpret features to explain model predictions.

### Phase 3: Result Cleanup & Extraction (~1/2 week)
- Final documentation of analytical findings in a Jupyter notebook.
- Detailed interpretations of statistical analyses, model results, and insights.
- Visualization of key findings and anomalies (charts, heatmaps, confusion matrices).

### Phase 4: Communication (~1/2 week)
- Preparation of a concise two-page report summarizing essential insights, significance of findings, and actionable recommendations in AAAI two-column format.
- Creation of a clear, engaging live presentation for stakeholders and the class.

## Project Deliverables

- **Jupyter Notebook:** Clearly documented analytical processes, model development, and insights.
- **Two-Page Report:** Concise summary of the project's findings, significance, and recommendations.
- **Presentation:** Effective, stakeholder-oriented communication of the project's outcomes.

## Grading Criteria

Projects will be evaluated based on:
- Alignment with customer objectives and clarity of the analytical purpose.
- Statistical and methodological soundness.
- Clarity, comprehensibility, and professional communication in notebook, written report, and presentation.
- Actionable recommendations clearly connected to data-driven findings.

## Customer Interaction
- Representative: **Adam West**
- Next Meeting: **Friday, 04/04/2025** (Teams)
- Purpose: Clarification, Q&A, data insights discussion

This structured approach ensures thorough investigation, predictive modeling, and actionable insights tailored to the school district's specific needs.

