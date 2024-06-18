from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #support vector classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()
X = data.data
Y = data.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    #random state of 23 makes sure to keep same values

#classifing svc
clf = SVC(kernel='linear', C=3)
    #kernel would be linear and soft margin is 3
clf.fit(x_train, y_train)

#classifing K-Neighbors
clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)

#Decision Tree classifier
clf3 = DecisionTreeClassifier()
clf3.fit(x_train, y_train)

#RandomForest classifier
clf4 = RandomForestClassifier()
clf4.fit(x_train, y_train)

#Compare score btw diff classifiers
print(f'SVC: {clf.score(x_test, y_test)}') #score of SVM = 96%
print(f'KNN: {clf2.score(x_test, y_test)}') #score of K = 96%
print(f'DTC: {clf3.score(x_test, y_test)}') #93%
print(f'RFC: {clf4.score(x_test, y_test)}') #94%
#Scores are roughly the same but SVM are generally better
#generally same performance in this case