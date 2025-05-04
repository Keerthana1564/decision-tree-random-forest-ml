import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
df = pd.read_csv("heart.csv")  
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree (max_depth=4)")
plt.show()
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title="Feature Importances (Random Forest)")
plt.show()
tree_scores = cross_val_score(tree_clf, X, y, cv=5)
rf_scores = cross_val_score(rf_clf, X, y, cv=5)
print("Decision Tree CV Accuracy:", tree_scores.mean())
print("Random Forest CV Accuracy:", rf_scores.mean())
