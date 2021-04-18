import numpy as np


class Nearest_Neighbour:
    def __init__(self):
        pass

    def train(self, X, Y): #Memorize training data O(1)
        # X is N x D where each row is an example. Y is 1-dimenstion of size N
        self.Xtr = X
        self.Ytr = Y

    def predict(self, X): # O(N)
        num_test = X.shape[0]
        # To make sure the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)

        for i in range(num_test): # For each test image : Find the closest train image and predict label of the nearest image
            # Uses manhattan distance to finde the nearest training image to the ith test image
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances) # Get index with the smallest distance
            Ypred[i] = self.Ytr[min_index] # Predict the label of the nearest example

        return Ypred
