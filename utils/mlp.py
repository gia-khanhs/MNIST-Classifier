import numpy as np

from .mnistData import mnist

class mlp:
    
    def __init__(self):
        self.W1 = np.random.random((10, 784)) - 0.5
        self.b1 = np.random.random((10, 1)) - 0.5
        self.W2 = np.random.random((10, 10)) - 0.5
        self.b2 = np.random.random((10, 1)) - 0.5

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        self.L = None

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z)
        denominator = np.sum(expZ, axis=0, keepdims=True)
        return expZ / denominator

    def forthProp(self):
        self.Z1 = np.dot(self.W1, mnist.train.img.T) + self.b1
        self.A1 = self.ReLU(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)

        # self.L = 

    def backProp(self):
        dZ2 = self.A2 - mnist.train.oneHotLabel # cross-entropy + softmax

        dW2 = np.dot(dZ2, self.A1.T) / mnist.train.size
        db2 = np.sum(dZ2, axis=1, keepdims=True) / mnist.train.size

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (self.Z1 > 0)

        dW1 = np.dot(dZ1, mnist.train.img) / mnist.train.size # the images are displayed in rows of the matrix => no need to tranpose
        db1 = np.sum(dZ1, axis=1, keepdims=True) / mnist.train.size

        return dW1, db1, dW2, db2
    
    def updateParams(self, learningRate, dW1, db1, dW2, db2):
        self.W2 -= learningRate * dW2
        self.b2 -= learningRate * db2
        self.W1 -= learningRate * dW1
        self.b1 -= learningRate * db1
    
    def gradientDescent(self, learningRate):
        self.forthProp()
        dW1, db1, dW2, db2 = self.backProp()
        self.updateParams(learningRate, dW1, db1, dW2, db2)

    #=================================================================================

    def calcTrainAccuracy(self):
        predictions = np.argmax(self.A2, axis=0, keepdims=True)
        accuracy = np.sum(mnist.train.label == predictions) / mnist.train.size
        
        return accuracy

    def predict(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)
        predictions = np.argmax(A2, axis=0, keepdims=True)
        
        return predictions

    def calcTestAccuracy(self):
        predictions = self.predict(mnist.test.img.T)
        accuracy = np.sum(mnist.test.label == predictions) / mnist.test.size

        return accuracy

    def train(self, learningRate, iterations, dPrint = 10):
        for i in range(1, iterations + 1):
            self.gradientDescent(learningRate)
            

            if i == 1 or i % dPrint == 0:
                accuracy = self.calcTrainAccuracy()
                print(f"Iteration #{i} | Training set Accuracy: {accuracy}")

        print("\nTraining complete!")
        accuracy = self.calcTrainAccuracy()
        print(f"Accuracy on training set: {accuracy}")
        accuracy = self.calcTestAccuracy()
        print(f"Accuracy on test set: {accuracy}")

    #=================================================================================

    def saveParams(self):
        np.savez("saves/savedParameters.npz", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def loadParams(self):
        loadedParams = np.load("saves/savedParameters.npz")
        self.W1 = loadedParams["W1"]
        self.b1 = loadedParams["b1"]
        self.W2 = loadedParams["W2"]
        self.b2 = loadedParams["b2"]