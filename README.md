# Supervised Machine Learning Classifier: Titanic Survival Prediction

An end-to-end predictive machine learning script using **Scikit-Learn** and **Python** to build, evaluate, and tune a binary Decision Tree Classifier using historical demographic metrics from the Titanic dataset.

## 📊 Project Pipelines & Checkpoints
This repository fulfills all 8 sequential engineering milestones required for the core supervised classification criteria:
1. **Prediction Definition:** Formulating an explicit target-vs-feature objective sentence.
2. **Data Cleansing:** Pruning missing data values and transforming categorical strings into numeric binary flags.
3. **Stratified Splitting:** Partitioning data into independent 80% training and 20% validation sets with proportional target cross-distributions.
4. **Baseline Benchmarking:** Calculating default baseline accuracy thresholds to ensure true predictive utility.
5. **Multi-Metric Evaluation:** Diagnostic profiling using raw Accuracy scores alongside complete Confusion Matrix quadrant mapping.
6. **Overfitting Diagnostics:** Evaluating hyperparameter bounds across `max_depth` ranges from 1 to 12.

## 📈 Executive Key Findings Summary
* **Model vs. Baseline:** The optimization pipeline achieved a **78.3% classification test accuracy**, soundly outperforming the passive guessing baseline score by approximately **~17%**.
* **Primary Prediction Drivers:** Feature weight outputs establish **passenger sex** as the absolute dominant structural vector determining survival priority splits.
* **Tuning Thresholds:** Multi-depth tracking proves that restricting the model layout to a range of **max_depth=3 or 4** maximizes production generalization capability before over-memorization variance sets in.

## 🚀 Repository Contents
* `machine_learning_classifier.py` - Production pipeline code executing data loading, encoding, stratified splitting, model fitting, and metrics tracing.
* `overfitting_depth_analysis.png` - Compiled high-resolution tree depth optimization curve visualization chart.
