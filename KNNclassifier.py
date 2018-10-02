class KNN_():
    def train( self, x_train, y_train ):
        self.x_train = x_train
        self.y_train = y_train
        return
    
    def predict_one(self, x_test, k):
        distances = []
        for i in range(len(self.x_train)):
            distance = ((self.x_train[i, :] - x_test)**2).sum()
            distances.append([distance, i])
        distances = sorted(distances)
        targets = []
        for i in range(k):
            index_of_training_data = distances[i][1]
            targets.append(self.y_train[index_of_training_data])
        return Counter(targets).most_common(1)[0][0] 
    
    def predict(self,x_test_data, k):
        predictions = []
        for x_test in x_test_data:
            predictions.append(self.predict_one(x_test, k))
        return predictions


from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

iris = datasets.load_iris()   #load dataset

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 0)

# make object of KNN class
classifier = KNN_()
classifier.train(X_train,Y_train)
Y_pred = classifier.predict( X_test, 7)
print(Y_pred)

# finding accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))