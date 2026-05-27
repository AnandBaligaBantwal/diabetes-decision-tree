## 📊 Project Overview
An experimental data science project evaluating how data constraints and class imbalances affect a `DecisionTreeClassifier` attempting to diagnose Type-2 diabetes using the PIMA Indians dataset.

## 🛠️ Key Work Done
* **Foundational Math:** Built custom `Entropy` and `Gini Impurity` equations from scratch to manually map feature disorder before training.
* **Controlled Experiments:** Slicing the dataset into low-dimensional feature subsets to run baseline tests (Sub-models A & B), proving that limiting attributes severely cripples model intelligence.
* **The Recall Rescue (Model C):** Engineered a final full-feature model using all 8 clinical variables and applied `class_weight="balanced"` to successfully eliminate major algorithmic bias.

## 📈 Model Evaluation (Diabetic Class F1-Score)
* **Model C (Full Feature + Balanced):** **0.65** 🏆 *(Precision: 0.64 / Recall: 0.65)*
* **Models A & B (Constrained):** **Failed** *(Suffered heavily from class bias with an unacceptable 0.41–0.50 recall)*

## 🎯 Significance
* **Medically Defensive Modeling:** Prioritizes high Recall (0.65) over blind overall accuracy, minimizing dangerous False Negatives where a sick patient might otherwise be sent home undiagnosed.
* **Methodological Storytelling:** Provides a clear progression script that demonstrates how to identify data bottlenecks, address class imbalances, and optimize a baseline tree framework.
