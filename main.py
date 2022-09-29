import numpy as np
import operator
from operator import itemgetter
import sklearn.datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score




def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, K=3):
        self.K = K

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(),
                                        key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0])
        return predictions

if __name__ == '__main__':
    mnist = sklearn.datasets.load_digits()

    #print(mnist.data.shape) #(1797, 64)
    #print(type(mnist)) #<class 'sklearn.utils._bunch.Bunch'>
    #print(mnist.feature_names) #['pixel_0_0', 'pixel_0_1', 'pixel_0_2'...

    X = mnist.data
    y = mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

   # print(X_train.shape, y_train.shape)  #(1347, 64) (1347,)
   # print(X_test.shape, y_test.shape)  # (450, 64) (450,)
   # print(np.unique(y_train, return_counts=True))
   # print(np.unique(y_test, return_counts=True))
    model = KNN(K=5)

    kVals = np.arange(3, 20, 2)
    accuracies = []
    for k in kVals:
        model = KNN(K=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accuracies.append(acc)
        print("K = " + str(k) + "; Accuracy: " + str(acc))

    dump(model, 'modelKNN.joblib')




   #df = pd.DataFrame(mnist)
    # df.head()