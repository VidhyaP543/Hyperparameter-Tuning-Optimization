# Hyperparameter Tuning and Optimization Report

## Objective
To improve machine learning model performance by tuning hyperparameters using Grid Search and Randomized Search techniques.

---

## Dataset Used
- Iris Dataset (Scikit-learn)
- Classification problem with 3 classes

---

## Model Used
- Random Forest Classifier

---

## Hyperparameter Tuning Methods

### 1. Grid Search
- Tries all possible combinations of parameters
- More accurate but slower

Parameters used:
- n_estimators: [50, 100]
- max_depth: [3, 5, 10]
- min_samples_split: [2, 5]

---

### 2. Randomized Search
- Randomly selects combinations
- Faster and more scalable

Parameters used:
- n_estimators: 50 to 150
- max_depth: [3, 5, 10, None]
- min_samples_split: [2, 5, 10]

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Cross-validation (5-fold)

---

## Results Summary

| Method | Best Parameters | Accuracy |
|--------|----------------|----------|
| Grid Search | (auto-filled from output) | XX.XX |
| Random Search | (auto-filled from output) | XX.XX |

---

## Conclusion
Both Grid Search and Randomized Search improved model performance by finding optimal hyperparameters. Randomized Search is faster, while Grid Search gives more precise tuning.

---

## Learning Outcome
- Understood hyperparameters in ML models
- Learned Grid Search and Randomized Search
- Improved model performance using tuning techniques
