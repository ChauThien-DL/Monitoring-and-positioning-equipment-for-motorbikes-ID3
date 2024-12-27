import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn import tree
import matplotlib.pyplot as plt
from collections import Counter

file_path = 'test.csv'  
data = pd.read_csv(file_path)

X = data.drop(columns=['ID', 'Rules'])  
y = data['Rules'].astype(str)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def entropy(y):
    counts = Counter(y)
    probs = [count / len(y) for count in counts.values()]
    print("\nTính Entropy:")
    print(f"  Tần suất của các giá trị: {dict(counts)}")
    print(f"  Xác suất: {probs}")
    entropy_value = -sum(p * np.log2(p) for p in probs if p > 0)
    for p in probs:
        if p > 0:
            print(f"  -({p} * log2({p})) = {-p * np.log2(p)}")
    print(f"  Entropy = {entropy_value}")
    return entropy_value

def information_gain(X, y, feature):
    print(f"\nTính Information Gain cho đặc trưng '{feature}':")
    total_entropy = entropy(y)
    print(f"  Entropy ban đầu: {total_entropy}")
    
    feature_values = X[feature].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset_y = y[X[feature] == value]
        subset_entropy = entropy(subset_y)
        weight = len(subset_y) / len(y)
        weighted_entropy += weight * subset_entropy
        print(f"  Giá trị {value}:")
        print(f"    Số lượng: {len(subset_y)} / {len(y)} = {weight}")
        print(f"    Entropy của subset: {subset_entropy}")
        print(f"    Trọng số * Entropy = {weight} * {subset_entropy} = {weight * subset_entropy}")
    
    gain = total_entropy - weighted_entropy
    print(f"  Information Gain = {total_entropy} - {weighted_entropy} = {gain}")
    return gain

def feature_entropy(X, y, feature):
    feature_values = X[feature].unique()
    entropy_per_feature = {}
    
    for value in feature_values:
        subset_y = y[X[feature] == value]
        entropy_per_feature[value] = entropy(subset_y)
    
    return entropy_per_feature

for feature in X.columns:
    gain = information_gain(X_train, y_train, feature)
    print(f"Information Gain cho '{feature}': {gain}")
    
    feature_entropy_values = feature_entropy(X_train, y_train, feature)
    print(f"Entropy for {feature}:")
    for value, ent in feature_entropy_values.items():
        print(f"  Value {value}: {ent}")

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
    tree.plot_tree(decision_tree, feature_names=feature_names, filled=False, rounded=False, label='root', impurity=False)
    plt.title("Simple Decision Tree Visualization")
    plt.show()

plot_simple_tree(model, feature_names=X.columns)

model_path = 'decision_tree_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
