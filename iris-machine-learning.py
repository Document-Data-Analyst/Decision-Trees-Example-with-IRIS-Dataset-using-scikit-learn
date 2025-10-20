from sklearn.datasets import load_iris  #  Load iris flower dataset
iris = load_iris()  #assign dataset to a variable
print("Data labels (target names)")
print("Labels are: {}".format(list(iris.target_names)) )   #print labels (target names)
print()

from sklearn import tree    # import the decision tree model
classifier = tree.DecisionTreeClassifier()      # Build a classifier

# Build the decision tree using the fit function
classifier = classifier.fit(iris.data, iris.target)

# Make predictions with petal data from dataset.
print("Petal Data: 5.1,3.5,1.4, 1.5")
print("Predict Label: {}".format(classifier.predict([[5.1,3.5,1.4, 1.5]])))     # 1, versicolor
print()
print("Petal Data: 5.1,3.5,1.4, 0.2")
print("Predict Label:  {}".format(classifier.predict([[5.1,3.5,1.4, 0.2]])))    # 0, setosa

