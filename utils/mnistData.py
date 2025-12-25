import numpy as np
import struct

'''
First 4 bytes: should be 2051
Next 4: number of images (nData)
Next 4: number of rows
Next 4: number of columns
The rest: data for each image (28 * 28 = 784 bytes for each)
'''

dataPath = "data/"

trainImgPath = dataPath + "train-images.idx3-ubyte"
trainLabelPath = dataPath + "train-labels.idx1-ubyte"

testImgPath = dataPath + "t10k-images.idx3-ubyte"
testLabelPath = dataPath + "t10k-labels.idx1-ubyte"

def readImages(filePath):
    magicNumber = None
    nData = None
    nRow = None
    nCol = None
    img = None

    try: 
        with open(filePath, 'rb') as file:
            magicNumber, nData, nRow, nCol = struct.unpack(">IIII", file.read(16))

            if magicNumber != 2051:
                print("Expected magic number: 2051. Please make sure the file contains images!")

            img = np.fromfile(file, dtype=np.uint8)
            img = img.reshape(nData, nRow * nCol)

            file.close()

    except FileNotFoundError:
        print(f"Error: The file {filePath} was not found!")

    return img


def readLabels(filePath):
    magicNumber = None
    nData = None
    nRow = None
    nCol = None
    label = None

    try:
        with open(filePath, 'rb') as file:
            magicNumber, nData = struct.unpack(">II", file.read(8))

            if magicNumber != 2049:
                print("Expected magic number: 2049. Please make sure the file contains labels!")

            label = np.fromfile(file, dtype=np.uint8)

            file.close()

    except FileNotFoundError:
        print(f"Error: The file {filePath} was not found!")

    return label

class mnist:
    class train:
        img = readImages(trainImgPath)
        label = readLabels(trainLabelPath)
        size = len(label)

    class test:
        img = readImages(testImgPath)
        label = readLabels(testLabelPath)
        size = len(label)
