import numpy as np
from utils.mlp import mlp

mnistClassifier = mlp()
mnistClassifier.train(learningRate=0.3, iterations=300)