import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_csv('data/processed/extracted_features_claude.csv')


X = df.drop(columns=['label'])
y = df['label']

# 4. Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Split the data (80% train, 20% test)
# Stratify is disabled here due to small sample sizes in specific classes
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Scaling (Critical for SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Define SVM and Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
}

# OvR is the default for SVC in multiclass scenarios
svm = SVC(kernel='rbf', probability=True, decision_function_shape='ovr', random_state=42)

# 8. Run GridSearch
# cv=2 is used due to the limited number of samples per class
grid = GridSearchCV(svm, param_grid, refit=True, verbose=1, cv=2)
grid.fit(X_train_scaled, y_train)

# 9. Output results for every combination
print("\n--- Grid Search Accuracy for each combination ---")
results_df = pd.DataFrame(grid.cv_results_)
for idx, row in results_df.iterrows():
    print(f"Params: {row['params']} | Mean CV Accuracy: {row['mean_test_score']:.4f}")

# 10. Final Evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(f"\nBest Parameters Found: {grid.best_params_}")
print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")