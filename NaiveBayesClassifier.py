def makeLabelled(column):
    second_limit = column.mean()
    first_limit = 0.5 * second_limit
    third_limit = 1.5*second_limit
    for i in range (0,len(column)):
        if (column[i] < first_limit):
            column[i] = 0
        elif (column[i] < second_limit):
            column[i] = 1
        elif(column[i] < third_limit):
            column[i] = 2
        else:
            column[i] = 3
    return column


class Naive_Bayes():
    def fit(self, X_train, Y_train):
        #dictionary to store frequency of data
        self.dictionary = {}
        class_values = set(Y_train)
        for current_class in class_values:
            self.dictionary[current_class] = {}
            self.dictionary["total_data"] = len(Y_train)
            current_class_rows = (Y_train == current_class)
            X_train_current = X_train[current_class_rows]
            Y_train_current = Y_train[current_class_rows]
            num_features = X_train.shape[1]
            self.dictionary[current_class]["total_count"] = len(Y_train_current)
            for j in range(1, num_features + 1):
                self.dictionary[current_class][j] = {}
                all_possible_values = set(X_train[:, j - 1])
                for current_value in all_possible_values:
                    self.dictionary[current_class][j][current_value] = (X_train_current[:, j - 1] == current_value).sum()
        return 
    
    def probability(self, x, current_class):
        output = np.log(self.dictionary[current_class]["total_count"]) - np.log(self.dictionary["total_data"])
        num_features = len(self.dictionary[current_class].keys()) - 1;
        for j in range(1, num_features + 1):
            xj = x[j - 1]
            count_current_class_with_value_xj = self.dictionary[current_class][j][xj] + 1
            count_current_class = self.dictionary[current_class]["total_count"] + len(self.dictionary[current_class][j].keys())
            current_xj_probablity = np.log(count_current_class_with_value_xj) - np.log(count_current_class)
            output = output + current_xj_probablity
        return output
    
    def predictSinglePoint(self, x):
        classes = self.dictionary.keys()
        best_p = -1000
        best_class = -1
        first_run = True
        for current_class in classes:
            if (current_class == "total_data"):
                continue
            p_current_class = self.probability(x, current_class)
            if (first_run or p_current_class > best_p):
                best_p = p_current_class
                best_class = current_class
            first_run = False
        return best_class
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            x_class = self.predictSinglePoint( x)
            y_pred.append(x_class)
        return y_pred
   

    
from sklearn import datasets
import numpy as np
#load data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#converting continuous data into labelled data
for i in range(0, X.shape[-1]):
    X[:, i] = makeLabelled(X[:, i])
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)

classifier = Naive_Bayes()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# finding accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))
