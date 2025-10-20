from sklearn.datasets import load_iris   # Load iris flower dataset
from sklearn import tree                 # Import decision tree model
import matplotlib.pyplot as plt          # Import Matplotlib for visualization

# Load the dataset
iris = load_iris()
print("Data labels (target names)")
print("Labels are: {}".format(list(iris.target_names)))
print()

# Build the decision tree
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(iris.data, iris.target)

# Make predictions
print("Petal Data: 5.1,3.5,1.4, 1.5")
print("Predict Label: {}".format(classifier.predict([[5.1, 3.5, 1.4, 1.5]])))
print()
print("Petal Data: 5.1,3.5,1.4, 0.2")
print("Predict Label: {}".format(classifier.predict([[5.1, 3.5, 1.4, 0.2]])))

# --- Visualization Section ---
plt.figure(figsize=(16, 10))
tree.plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Iris Dataset", fontsize=16)
plt.show()
