from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
gnb_proba = gnb.predict_proba(X_test)[:, 1]
print("=== GaussianNB ===")
print(classification_report(y_test, gnb_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, gnb_pred))
print("ROC AUC:", roc_auc_score(y_test, gnb_proba))
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True, random_state=42))
])
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.001]
}
grid = GridSearchCV(
    svm_pipe, param_grid, cv=5, n_jobs=-1, scoring="roc_auc"
)
grid.fit(X_train, y_train)
best_svm = grid.best_estimator_
svm_pred = best_svm.predict(X_test)
svm_proba = best_svm.predict_proba(X_test)[:, 1]
print("\n=== SVM (RBF) ===")
print("Best Params:", grid.best_params_)
print(classification_report(y_test, svm_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("ROC AUC:", roc_auc_score(y_test, svm_proba))
