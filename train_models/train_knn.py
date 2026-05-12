import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Helper function to parse the string arrays in the CSV
def parse_flex_string(s):
    # Remove brackets and newlines, then split by whitespace
    s = s.replace('[', '').replace(']', '').replace('\n', ' ')
    return np.fromstring(s, sep=' ')

def univariate_dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / (n + m)

# 2. Custom Distance Metric for KNN
def custom_dtw_combined_dist(x1, x2):
    """
    Custom distance metric:
    - First 320 elements are 8 flex sensors (40 each)
    - Last 14 elements are pad z and r values
    """
    total_dist = 0
    # DTW for each of the 8 flex sensors
    for i in range(8):
        s1 = x1[i*40 : (i+1)*40]
        s2 = x2[i*40 : (i+1)*40]
        total_dist += univariate_dtw_distance(s1, s2)
    
    # Euclidean distance for the pad features
    pad1 = x1[320:]
    pad2 = x2[320:]
    total_dist += np.linalg.norm(pad1 - pad2)
    
    return total_dist

# 3. Load and Preprocess Data
df = pd.read_csv('extracted_features_pipeline_2.csv')

# Convert flex columns from strings to lists of arrays
flex_cols = [f'flex{i}' for i in range(8, 16)]
pad_cols = [col for col in df.columns if 'pad' in col]

processed_rows = []
for idx, row in df.iterrows():
    # Flatten all flex sensors into one array
    flex_data = np.concatenate([parse_flex_string(row[col]) for col in flex_cols])
    # Append pad features
    pad_data = row[pad_cols].values.astype(float)
    combined_features = np.concatenate([flex_data, pad_data])
    processed_rows.append(combined_features)

X = np.array(processed_rows)
y = df['label']

# 4. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Scaling
# We scale the entire vector to ensure pad features and flex magnitudes are comparable
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Define KNN with Custom Metric and Grid Search
# Note: 'brute' algorithm is required for custom callable metrics
knn_param_grid = {
    'n_neighbors': [1, 2, 3, 5, 10, 15],
    'weights': ['uniform', 'distance']
}

# Initialize KNN with the custom DTW distance function
knn_base = KNeighborsClassifier(metric=custom_dtw_combined_dist, algorithm='brute')

# Grid Search
# n_jobs is set to 1 because custom python metrics often have serialization overhead
grid_search = GridSearchCV(knn_base, knn_param_grid, cv=2, scoring="accuracy", n_jobs=1)
grid_search.fit(X_train_scaled, y_train)

# 8. Output results
print("\n--- Grid Search Accuracy for each combination ---")
results_df = pd.DataFrame(grid_search.cv_results_)
for idx, row in results_df.iterrows():
    print(f"Params: {row['params']} | Mean CV Accuracy: {row['mean_test_score']:.4f}")

# 9. Final Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(f"\nBest Parameters Found: {grid_search.best_params_}")
print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")