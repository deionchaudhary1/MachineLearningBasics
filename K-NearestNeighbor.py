#using breastcancer dataset from scikit-learn
    #classifies tumors as bad or good
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

#take look on data
data = load_breast_cancer()
print(data.feature_names) #different data to classify tumor as malignant or benign
print(data.target_names) #two classifications

#split into train and test
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size = 0.2)
#data.data - features, data.target is classifications

#classifier and train
clf = KNeighborsClassifier(n_neighbors=3) #how many neighbors we want to look at
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test)) #score on testing the model

#clf.predict([]) --> predict on picked 