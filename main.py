import numpy as np
import matplotlib

from utils.mnistData import mnist

def initParams():
    W1 = np.random.random((10, 784)) - 0.5
    b1 = np.random.random((10, 1)) - 0.5
    W2 = np.random.random((10, 10)) - 0.5
    b2 = np.random.random((10, 1)) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    normalise = np.max(Z, axis=0, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

def forthProp(W1, b1, W2, b2):
    Z1 = np.dot(W1, mnist.train.img.T) + b1
    A1 = ReLU(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY

def backProp(Z1, A1, Z2, A2, W1, W2):
    oneHotY = oneHot(mnist.train.label)
    
    # L = crossEntropy(oneHotY, A2)
    
    # dA2 = L / A2
    # dA2/dZ2 = jacobian stuff
    # https://www.youtube.com/watch?v=rf4WF-5y8uY
    dZ2 = A2 - oneHotY #dL / dZ2. the resulting derivative of softmax + cross-entropy (dark magic)
    
    # Z2 = W2 * A1 + b2
    # dZ2/dW2 = A1
    # dL/dW2 = dL/dZ2 * dZ2/dW2
    dW2 = np.dot(dZ2, A1.T) / mnist.train.size
    # dZ2/db2 = 1
    # dL/db2 = dL/dZ2 * 1
    db2 = np.sum(dZ2, axis=1, keepdims=True) / mnist.train.size

    # Z2 = W2 * A1 + b2 = W2 * ReLU(Z1) + b2
    # dZ2/dA1 = W2
    # dL/dA1 = dL/dZ2 * dZ2/dA1
    dA1 = np.dot(W2.T, dZ2)
    # dA1/dZ1 = ReLu'(Z1)
    # dL/dZ1 = dL/dA1 * dA1/dZ1
    dZ1 = dA1 * (Z1 > 0)

    dW1 = np.dot(dZ1, mnist.train.img) / mnist.train.size
    db1 = np.sum(dZ1, axis=1, keepdims=True) / mnist.train.size

    return dW1, db1, dW2, db2

def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 -= learningRate * dW1
    b1 -= learningRate * db1
    W2 -= learningRate * dW2
    b2 -= learningRate * db2

    return W1, b1, W2, b2

def getPrediction(A2):
    return np.argmax(A2, axis=0)

def getAccuracy(predictions):
    return np.sum(predictions == mnist.train.label) / mnist.train.size

def gradientDescent(learningRate, iterations):
    W1, b1, W2, b2 = initParams()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forthProp(W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backProp(Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate)

        if(i % 10 == 0):
            predictions = getPrediction(A2)
            print(f"Iteration: {i}   |   Accuracy: {getAccuracy(predictions)}")

    return W1, b1, W2, b2

def predictTest(newImg, W1, b1, W2, b2):
    Z1 = np.dot(W1, newImg.T) + b1
    A1 = ReLU(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return np.argmax(A2, axis=0)

def testAccuracy(W1, b1, W2, b2):
    predictions = predictTest(mnist.test.img, W1, b1, W2, b2)
    accuracy = np.sum(predictions == mnist.test.label) / mnist.test.size
    print(f"Accuracy on test set: {accuracy}")

W1, b1, W2, b2 = gradientDescent(0.3, 301)
testAccuracy(W1, b1, W2, b2)