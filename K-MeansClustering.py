from sklearn.cluster import k_means
from sklearn.preprocessing import scale #we use this normalize the data (0-1)
from sklearn.datasets import load_digits
    #pictures of handwriten digits (in our case we dont label them but we sort them into clusters by pattern recognition)
    #program doesn't know the number

digits = load_digits()
data = scale(digits.data)

#K-means - model trains
model = k_means(n_clusters=10, init='random', n_init=10)
    #10 clusters, random placing of centroids and 10 centriods
model.fit(data)

#Doesn't make sense to test since we don't have a sense of right and wrong
model.predict([...]) #add in a picture
    #only measure of accuracy is seeing how well the centroids are placed
