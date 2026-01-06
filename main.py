import numpy as np 

from utils.mlp import mlp

#=====================================================
training = False

mnistClassifier = mlp()
if training:
    mnistClassifier.train(learningRate=0.3, iterations=300)
    mnistClassifier.saveParams()

    print("Parameters saved!")

else:
    mnistClassifier.loadParams()

    accuracy = mnistClassifier.calcTestAccuracy()
    print("Parameters loaded!")
    print(f"Accuracy on test set: {accuracy}")

#=====================================================