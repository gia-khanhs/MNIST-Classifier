import numpy as np
import matplotlib

from utils.mnistData import mnist

def initParams():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    normalise = np.max(Z, axis=0, keepdims=True)
    return np.exp(Z - normalise) / np.sum(np.exp(Z - normalise), axis=0, keepdims=True)

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

def backProp(Z1, A1, Z2, A2):
    oneHotY = oneHot(mnist.train.label)
    
    # L = np.maximum(Z2, oneHotY) - np.minimum(Z2, oneHotY)
    # L = np.sum(L, axis=0, keepdims=True)

    dZ2 = A2 - oneHotY #dL / dZ2

    return None

W1, b1, W2, b2 = initParams()
Z1, A1, Z2, A2 = forthProp(W1, b1, W2, b2)
backProp(Z1, A1, Z2, A2)
