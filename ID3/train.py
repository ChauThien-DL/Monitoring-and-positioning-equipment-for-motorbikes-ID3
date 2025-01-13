import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn import tree
import matplotlib.pyplot as plt
from collections import Counter

file_path = 'hihi.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['ID', 'Rules'])  
y = data['Rules'].astype(str)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_data = X_train.copy()
train_data['Rules'] = y_train

test_data = X_test.copy()
test_data['Rules'] = y_test
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

def entropy(y):
    counts = Counter(y)
    probs = [count / len(y) for count in counts.values()]
    entropy_value = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropy_value

def information_gain(X, y, feature):
    total_entropy = entropy(y)
    feature_values = X[feature].unique()
    weighted_entropy = 0

    for value in feature_values:
        subset_y = y[X[feature] == value]
        weight = len(subset_y) / len(y)
        subset_entropy = entropy(subset_y)
        weighted_entropy += weight * subset_entropy

    gain = total_entropy - weighted_entropy
    return gain

target_entropy = entropy(y_train)
print(f"Entropy của cột Rules: {target_entropy}")

for feature in X.columns:
    gain = information_gain(X_train, y_train, feature)
    print(f"Information Gain cho đặc trưng '{feature}': {gain}")

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

def plot_simple_tree(decision_tree, feature_names):
    tree_rules = tree.export_text(decision_tree, feature_names=feature_names)
    print(tree_rules)
    plt.figure(figsize=(20, 5))
    tree.plot_tree(decision_tree, feature_names=feature_names, filled=True, rounded=True)
    plt.title("Simple Decision Tree Visualization")
    plt.show()

plot_simple_tree(model, feature_names=X.columns)

model_path = 'decision_tree_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

