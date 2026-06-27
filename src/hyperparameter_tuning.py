from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base model
rf = RandomForestClassifier(random_state=42,n_estimators=50)

# ---------------------------
# 1. GRID SEARCH
# ---------------------------
grid_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=grid_params,
    cv=3,
    n_jobs=1
)

grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_grid_model = grid_search.best_estimator_
grid_pred = best_grid_model.predict(X_test)

print("\nGRID SEARCH RESULTS")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, grid_pred))
print(classification_report(y_test, grid_pred))

# ---------------------------
# 2. RANDOMIZED SEARCH
# ---------------------------
random_params = {
    'n_estimators': np.arange(50, 200, 50),
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_params,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=1
)

random_search.fit(X_train, y_train)

# Best model from Random Search
best_random_model = random_search.best_estimator_
random_pred = best_random_model.predict(X_test)

print("\n RANDOM SEARCH RESULTS")
print("Best Parameters:", random_search.best_params_)
print("Accuracy:", accuracy_score(y_test, random_pred))
print(classification_report(y_test, random_pred))
