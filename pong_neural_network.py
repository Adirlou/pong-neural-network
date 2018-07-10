import sys
import random
import numpy as np
from pong_state import *

"""
Reads the input file containing the expert policy
"""
def readData(fileName):
    data = []
    with open(fileName, "r") as f:
        line = f.readline()
        while line:
            l = line.split(" ")
            if len(l) != 6:
                print("cannot read line : " + line)
            for i, e in enumerate(l):
                l[i] = float(e.strip())
            t = tuple(l)
            data.append(t)
            line = f.readline()
    return data

"""
Compute the softmax of an array
"""
def softmax(x):
    return np.exp(x) / np.exp(x).sum()

"""
Compute the cross entropy loss function
"""
def crossEntropy(F, y):
    batchSize = len(F)
    lossFunc = 0

    for i in range(batchSize):
        temp = np.exp(F[i]).sum()
        lossFunc += F[i, y[i]] - np.log(temp)
    lossFunc = -(1.0/batchSize)*lossFunc

    dF = np.zeros(F.shape)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            jEqualsYi = 1 if j == y[i] else 0
            dF[i, j] =-(1.0/batchSize)*(jEqualsYi - softmax(F[i])[j])

    return (lossFunc, dF)

"""
Performs the mini-batch gradient descent algorithm to approximate the optimal weights
"""
def minibatchGD(data, epoch, batchSize, d, dPrime, weightScale, learningRate=0.1):
    #original test data
    testX = np.array([x[:-1] for x in data])
    testy = np.array([int(x[-1]) for x in data])

    lossArray = []
    accuracyArray = []

    #Initialize random weights and bias
    W1 = weightScale * np.random.rand(d, dPrime)
    W2 = weightScale * np.random.rand(dPrime, dPrime)
    W3 = weightScale * np.random.rand(dPrime, dPrime)
    W4 = weightScale * np.random.rand(dPrime, 3)
    Ws = [W1, W2, W3, W4]

    b1 = np.zeros(dPrime)
    b2 = np.zeros(dPrime)
    b3 = np.zeros(dPrime)
    b4 = np.zeros(3)
    bs = [b1, b2, b3, b4]

    #Repeat for a certain number of epochs
    for e in range(epoch):
        print("Epoch " + str(e))
        np.random.shuffle(data)
        for i in range(0, len(data), batchSize):
            sys.stdout.write("\rBatch " + str(i/batchSize) + "/" + str(len(data) / batchSize))
            sys.stdout.flush()
            batch = data[i: i + batchSize]
            X = np.array([x[:-1] for x in batch])
            y = np.array([int(x[-1]) for x in batch])

            loss = fourLayerNetworkTrain(X, Ws, bs, y, learningRate)
        print("")
        print("Loss at epoch "+str(e)+": "+str(loss))
        lossArray.append(loss)

        accuracy = computeAccuracy(testX, Ws, bs, testy)
        print("Accuracy at epoch "+str(e)+": "+str(accuracy))
        accuracyArray.append(accuracy)
        print("")

    return (Ws, bs, lossArray, accuracyArray)

"""
Compute the accuracy of the network on the training data
"""
def computeAccuracy(data, Ws, bs, y):
    output = fourLayerNetworkPredict(data, Ws, bs)[3]
    counter = 0.0
    for i in range(len(output)):
        if output[i] == y[i]:
            counter += 1
    return counter / len(output)


"""
Compute the confusion matrix
"""
def confusionMatrix(data, Ws, bs, y):
    conf = np.zeros((len(Action), len(Action)))
    output = fourNetworkPredict(data, Ws, bs)[3]
    for i in range(len(output)):
            conf[y[i], output[i]] += 1.0
    for i in range(len(conf)):
        conf[i] /= conf[i].sum()
    return conf

"""
Update the weights of the network
"""
def fourLayerNetworkTrain(X, Ws, bs, y, eta):
    layerCount = 4

    #First compute the output on input X
    (F, aCaches, rCaches, _) = fourLayerNetworkPredict(X, Ws, bs)

    #Compute loss and back propagate
    loss, dF = crossEntropy(F, y)
    dAs = [0.0] * layerCount
    dWs = [0.0] * layerCount
    dbs = [0.0] * layerCount
    dZ = dF
    for i in range(layerCount-1, -1, -1):
        dAs[i], dWs[i], dbs[i] = affineBackward(dZ, aCaches[i])
        if i == 0:
            break
        dZ = reluBackward(dAs[i], rCaches[i-1])

    for i, dW in enumerate(dWs):
        Ws[i] -= eta * dW
    for i, db in enumerate(dbs):
        bs[i] -= eta * db

    return loss

"""
Compute the output of the network on input X
"""
def fourLayerNetworkPredict(X, Ws, bs):
    layerCount = 4
    Zs = [0.0] * layerCount
    aCaches = [0.0] * layerCount
    rCaches = [0.0] * (layerCount - 1)
    A = X
    for i in range(layerCount):
        Zs[i], aCaches[i] = affineForward(A, Ws[i], bs[i])

        if i == layerCount - 1:
            break
        A, rCaches[i] = reluForward(Zs[i])

    F = Zs[-1]
    
    classifications = []
    for row in F:
        max_probability = row[0]
        max_label = 0
        for label, probability in enumerate(row):
            if probability > max_probability:
                max_probability = probability
                max_label = label
        classifications.append(max_label)

    return (F, aCaches, rCaches, classifications)

"""
Return the best action for the agent, given the output of the network
"""
def bestAction(state, Ws, bs):
    X = [state]
    classifications = fourLayerNetworkPredict(X, Ws, bs)[3]
    print(classifications)
    return classifications[0]


def affineForward(A, W, b):
    bT = b.transpose()
    temp = np.matmul(A, W)
    Z = np.add(temp, b)
    cache = (A, W, b)
    return (Z, cache)

def affineBackward(dZ, cache):
    A = cache[0]
    W = cache[1]
    B = cache[2]

    aT = A.transpose()
    wT = W.transpose()

    dA = np.matmul(dZ, wT)
    dW = np.matmul(aT, dZ)

    dB = np.zeros(shape=B.shape)

    for j in range(len(dZ[0])):
        for i in range(len(dZ)):
            dB[j] += dZ[i][j]

    return (dA, dW, dB)


def reluForward(z):
    newZ = np.zeros(z.shape)
    for i, zi in enumerate(z):
        for j, zij in enumerate(zi):
            newZij = zij
            if newZij <= 0:
                newZij = 0
            newZ[i,j] = newZij
    return (newZ, z)

def reluBackward(dA, zcache):
    dZ = np.zeros(zcache.shape)
    for i, zi in enumerate(zcache):
        for j, zij in enumerate(zi):
            if zij <= 0:
                dZ[i,j] = 0
            else:
                dZ[i,j] = dA[i,j]
    return dZ
