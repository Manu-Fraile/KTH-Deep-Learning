import math as m
from functions import *


def normalise(matrix):

    matrixMean = np.mean(matrix, axis=1)
    stdMatrix = np.std(matrix, axis=1)
    repetitions = len(matrix[0])

    matrixMean = np.transpose([matrixMean]*repetitions)
    stdMatrix = np.transpose([stdMatrix] * repetitions)

    matrix = matrix - matrixMean
    matrix = matrix / stdMatrix

    return matrix


def evaluateClassifier(X, W, b):

    # X (dx1)
    # W (Kxd)
    # b (Kx1)
    # p (Kx1)

    s = np.dot(W, X) + b
    p = softmax(s)

    return p


def ComputeCost(X, Y, W, b, lamda):

    sum1 = np.sum(crossentrpoyLoss(X, Y, W, b))
    sum2 = np.sum(W**2)
    D = X.shape[1]

    J = (sum1/D) + (lamda*sum2)

    return J


def crossentrpoyLoss(X, Y, W, b):

    p = evaluateClassifier(X, W, b)
    p_y = np.dot(Y.T, p)

    return (-1)*m.log(p_y, 10)


def ComputeAccuracy(X, Y, W, b):

    dataNum = trainX.shape[0]
    bingo = 0

    for i in range(trainX.shape[0]):
        P = evaluateClassifier(X[:, i], W, b)
        k_star = np.argmax(P)

        if k_star == Y[i]:
            bingo += 1

    acc = bingo/dataNum

    return acc


def miniBatchGD(X, Y, GDparams, W, b, lamda):

    J = ComputeCost(X, Y, W, b, lamda)
    Wstar, bstar = np.argmin(J)

    return Wstar, bstar

if __name__ == "__main__":

    # Take care with transpose or not
    # Take care with Y that goes from 0,...,9 and not 1,...,10
    training = LoadBatch('cifar-10-batches-py/data_batch_1')
    trainX = normalise(training[b'data'].T.astype(int))
    trainY = [int(i) for i in training[b'labels']]

    validation = LoadBatch('cifar-10-batches-py/data_batch_2')
    validX = normalise(validation[b'data'].T.astype(int))
    validY = [int(i) for i in validation[b'labels']]

    testing = LoadBatch('cifar-10-batches-py/test_batch')
    testX = normalise(testing[b'data'].T.astype(int))
    testY = [int(i) for i in testing[b'labels']]

    # Dimension parameters
    d = trainX.shape[1]
    n = trainX.shape[0]
    K = len(np.unique(trainY))

    # Initialise W (Kxd) and b (Kx1)
    mu = 0
    sigma = 0.01
    W = np.random.normal(mu, sigma, size=(K, d))
    b = np.random.normal(mu, sigma, size=(K, 1))

    # Control parameters. GDparams
    n_batch = 100
    eta = 0.001
    n_epochs = 20
    GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs}

    lamda = 0



    # P = evaluateClassifier(trainX[:, :100], W, b)
    #print(trainX.shape[1])