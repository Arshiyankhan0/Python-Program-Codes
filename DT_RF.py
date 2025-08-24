from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("=== Decision Tree ===")
print(classification_report(y_test, dt_pred, target_names=iris.target_names))
plt.figure(figsize=(10,6))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True)
plt.show()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("=== Random Forest ===")
print(classification_report(y_test, rf_pred, target_names=iris.target_names))
