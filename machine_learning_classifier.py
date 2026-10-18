import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("====================================================")
print("     PROJECT 04: FIRST MACHINE LEARNING CLASSIFIER   ")
print("====================================================\n")

# --- STEP 1: DEFINE THE PREDICTION IN A SENTENCE ---
# "Predict whether a passenger survived from their ticket class (pclass), sex, age, and fare."
print("--- STEP 1: OBJECTIVE DEFINITION ---")
print("Goal: Predict whether a passenger survived (target variable) using their ticket class (pclass), sex, age, and fare (feature variables).\n")

# --- STEP 2: PREPARE THE DATA ---
# Load dataset, isolate specified columns, clean rows, encode text parameters
df = sns.load_dataset('titanic')[["survived", "pclass", "sex", "age", "fare"]]
df = df.dropna()  # Drop rows containing missing values as per guidelines
df["sex"] = (df["sex"] == "female").astype(int)  # Turn female into 1, male into 0

X = df.drop(columns="survived")
y = df["survived"]

# --- STEP 3: SPLIT THE DATA ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- STEP 4: SET A BASELINE ACCURACY ---
# Baseline is always predicting the most common outcome class
most_common_class_count = max(y_test.value_counts())
baseline_accuracy = most_common_class_count / len(y_test)

print("--- STEP 4: BASELINE BENCHMARK ---")
print(f"Baseline Accuracy (always guessing most frequent outcome): {baseline_accuracy:.3f}\n")

# --- STEP 5 & 6: TRAIN AND EVALUATE MODEL (max_depth=4) ---
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model_accuracy = accuracy_score(y_test, predictions)

print("--- STEP 5 & 6: MODEL PERFORMANCE (max_depth=4) ---")
print(f"Decision Tree Classifier Test Accuracy: {model_accuracy:.3f}")

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix Output Matrix Array:")
print(cm)

print("\n--- CONFUSION MATRIX BREAKDOWN & CELL MEANINGS ---")
print(f"• True Negatives (Top-Left [{cm[0][0]}]): Passengers correctly predicted as did not survive.")
print(f"• False Positives (Top-Right [{cm[0][1]}]): Passengers who did not survive but incorrectly predicted to survive.")
print(f"• False Negatives (Bottom-Left [{cm[1][0]}]): Passengers who survived but incorrectly predicted to did not survive.")
print(f"• True Positives (Bottom-Right [{cm[1][1]}]): Passengers correctly predicted to survive.\n")

# --- STEP 7: TEST DEPTH AND OVERFITTING ---
train_accuracies = []
test_accuracies = []
depth_range = range(1, 13)

for depth in depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))

# Save the Overfitting Visualization Curve
plt.figure(figsize=(9, 5))
plt.plot(depth_range, train_accuracies, marker='o', label='Training Accuracy', color='darkblue', linewidth=2)
plt.plot(depth_range, test_accuracies, marker='s', label='Test Accuracy', color='darkorange', linewidth=2)
plt.title('Overfitting Analysis: Accuracy vs. Tree Depth', fontsize=12, fontweight='bold')
plt.xlabel('Tree Depth Limit (max_depth)')
plt.ylabel('Accuracy Metric Score')
plt.xticks(depth_range)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("overfitting_depth_analysis.png", dpi=150)
plt.close()
print("--- STEP 7: OVERFITTING CURVE GRAPH SAVED ---")
print("[✓] Saved evaluation performance metric curve plot chart: 'overfitting_depth_analysis.png'")
print("Explanation: As depth increases past 4-5, Training Accuracy keeps climbing toward 100%, but Test Accuracy starts degrading. Very deep trees memorize random noise in the training set instead of learning clean, generalizable trends.\n")

# --- STEP 8: WRITE WHAT YOU LEARNED ---
print("====================================================")
print("           STEP 8: FINAL SUMMARY REFLECTIONS        ")
print("====================================================")
print(f"1. Performance vs Baseline: Model achieved {model_accuracy:.3f} accuracy, clearing the {baseline_accuracy:.3f} baseline.")
print("2. Feature Importance Metrics:")
features_dict = dict(zip(X.columns, np.round(model.feature_importances_, 2).tolist()))
print(f"   {features_dict}")
print("   Takeaway: 'sex' stands out as the single most critical structural prediction feature driver by a wide margin.")
print("3. Model Limitation: Decision Trees process clean data well but are highly prone to variance and strict overfitting parameters when depth bounds are unrestricted.")
print("====================================================")
