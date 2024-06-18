import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#helps split data into training and testing

#DATA
time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1, 1) #vertical
scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]). reshape(-1,1)
#need data in vertical format for scikitlearn

time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.3)
    #use 20% of data for testing model
    #randomly chooses

model = LinearRegression()
model.fit(time_train, score_train) #only making model on training data

print(model.score(time_test, score_test)) #we use these to test if for the given hr, it gives the correlating score
#prints the accuracy (x100 is the percentage)
#returns the distance from the line
#by increasing the testing size, we would get more accurate and consistent answers

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0, 70, 100).reshape(-1,1), model.predict(np.linspace(0, 70, 100).reshape(-1,1)), 'r')
plt.show()
