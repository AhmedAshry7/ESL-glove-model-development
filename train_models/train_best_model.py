import pandas as pd
import numpy as np
import joblib
import json
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
classes = le.classes_.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm = SVC(kernel='rbf', probability=True, decision_function_shape='ovr', random_state=42, C=10, gamma=0.01)
svm.fit(X_scaled, y_encoded)

#save the model
joblib.dump(svm, 'models/best_svm_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Best SVM model saved as 'best_svm_model.pkl'")

with open('data/processed/classes.json', 'w', encoding='utf-8') as f:
    json.dump(classes, f)

print(f"SVM trained and saved. Classes: {classes}")