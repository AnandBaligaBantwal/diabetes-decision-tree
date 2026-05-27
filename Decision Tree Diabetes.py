import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
from math import log2
from sklearn.metrics import classification_report, recall_score

data = pd.read_csv("diabetes.csv")
data.head()


def categorize_bp(bp):
    if bp < 80:
        return "Low Pressure"
    elif 80 <= bp <= 120:
        return "Normal"
    else:
        return "High Pressure"


data["BP_Category"] = data["BloodPressure"].apply(categorize_bp)

pregnancy_data = (
    data.groupby(["Pregnancies", "Outcome"])["Outcome"]
    .count()
    .reset_index(name="count")
)

# 2. Build and display the barplot with your new title
plt.figure(figsize=(12, 6))
sns.barplot(data=pregnancy_data, x="Pregnancies", y="count", hue="Outcome")
plt.title("Distribution of Diabetic Outcomes by Number of Pregnancies")
plt.xlabel("Number of Pregnancies")
plt.ylabel("Patient Count")
plt.show()


def calculate_entropy(column):
    counts = column.value_counts(normalize=True)
    entropy = -sum(p * log2(p) for p in counts if p > 0)
    return entropy


entropy_bp = calculate_entropy(data["BP_Category"])
print(f"Entropy of BP Category: {entropy_bp:.4f}")


def gini_index(y):
    if len(y) == 0:
        return 0
    p1 = np.mean(y)
    return 1 - (p1**2 + (1 - p1) ** 2)


# Function to calculate Gini Index for a feature
def gini_for_feature(data, feature, target="Outcome"):
    values = data[feature].unique()
    gini = 0
    for val in values:
        subset = data[data[feature] == val]
        weight = len(subset) / len(data)
        gini_subset = gini_index(subset[target])
        gini += weight * gini_subset
    return gini


# Calculate Gini Index for the specified columns
columns_to_check = ["BMI", "Insulin", "DiabetesPedigreeFunction", "Pregnancies"]
gini_scores = {col: gini_for_feature(data, col) for col in columns_to_check}
min_gini_column = min(gini_scores, key=gini_scores.get)
print(f"Feature with Minimum Gini Impurity: {min_gini_column}")

# --- Sub-Model A: Pregnancies & Blood Pressure ---
XA = data[["Pregnancies", "BloodPressure"]]
yA = data["Outcome"]

# Split train and test with split size 0.2 and random state 24
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    XA, yA, test_size=0.2, random_state=24
)

# Fit the model with random_state=8
model_A = DecisionTreeClassifier(random_state=8)
model_A.fit(X_train_A, y_train_A)

# Get predictions
y_pred_A = model_A.predict(X_test_A)

# Calculate confusion matrix
cm_A = confusion_matrix(y_test_A, y_pred_A)

# Calculate precision
precision_A = precision_score(y_test_A, y_pred_A)

print(f"Model A (Pregnancies & BP) Precision: {precision_A:.2f}")

# --- Sub-Model B: Insulin & BMI ---
XB = data[["Insulin", "BMI"]]
# Dependent feature: Outcome
yB = data["Outcome"]

# Split train and test with split size 0.2 and random state 24
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    XB, yB, test_size=0.2, random_state=24
)

# Fit the model with random_state=30
model_B = DecisionTreeClassifier(random_state=30)
model_B.fit(X_train_B, y_train_B)

# Get predictions
y_pred_B = model_B.predict(X_test_B)

# Calculate recall
recall_B = recall_score(y_test_B, y_pred_B)

print(f"Model B (Insulin & BMI) Recall: {recall_B:.2f}")

## --- Sub-Model C: Full Feature Set ---
XC = data.drop(columns=["Outcome", "BP_Category"])
yC = data["Outcome"]

# Split train and test with split size 0.2 and random state 42
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(
    XC, yC, test_size=0.2, random_state=42
)

# Fit the model with class_weight="balanced" and random_state=8
model_C = DecisionTreeClassifier(class_weight="balanced", random_state=8)
model_C.fit(X_train_C, y_train_C)

# Get predictions
y_pred_C = model_C.predict(X_test_C)

# Calculate precision and recall for Model C
precision_C = precision_score(y_test_C, y_pred_C)
recall_C = recall_score(y_test_C, y_pred_C)

print(f"Model C (All Features) Precision: {precision_C:.2f}")
print(f"Model C (All Features) Recall: {recall_C:.2f}")
print("\nFull Classification Report for Model C:")
print(classification_report(y_test_C, y_pred_C))
