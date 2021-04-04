import math as m
from functions import *

from numba import jit, cuda


def normalise(matrix):

    matrixMean = np.mean(matrix, axis=1)
    stdMatrix = np.std(matrix, axis=1)
    repetitions = len(matrix[0])

    matrixMean = np.transpose([matrixMean]*repetitions)
    stdMatrix = np.transpose([stdMatrix]*repetitions)

    matrix = matrix - matrixMean
    matrix = matrix / stdMatrix

    return matrix


def evaluateClassifier(X, W, b):
    from functions import softmax
    # X (dx1)
    # W (Kxd)
    # b (Kx1)
    # p (Kx1)

    if X.shape == (3072,):
        X = X.reshape(len(X), 1)

    #print(X.shape)
    #print(W.shape)

    # X = X.reshape(len(X), 1)

    s = np.dot(W, X) + b
    p = softmax(s)

    #print(p.shape)
    #print('\n\n')

    #print('El valor de p es: {}'.format(p))

    return p


def ComputeCost(X, Y, W, b, lamda):

    sum1 = np.sum(crossentrpoyLoss(X, Y, W, b))
    sum2 = np.sum(W**2)
    D = X.shape[1]

    J = (sum1/D) + (lamda*sum2)

    #print('WIP')
    #print(J)

    return J


def crossentrpoyLoss(X, Y, W, b):

    p = evaluateClassifier(X, W, b)

    if Y.shape == (10,):
        Y = Y.reshape(10, 1)

    #print(p.shape)
    #print(Y.shape)

    p_y = np.dot(Y.T, p)

    return (-1)*(np.sum(Y*np.log(p)))#m.log(p_y, 10)


def ComputeAccuracy(X, y, W, b):

    dataNum = X.shape[1]
    bingo = 0

    for i in range(X.shape[1]):
        P = evaluateClassifier(X[:, i], W, b)
        k_star = np.argmax(P)

        #print(k_star)
        #print(y[i])
        #print('\n\n')

        if k_star == y[i]:
            bingo += 1

    acc = bingo/dataNum

    return acc


def ComputeGradients(X, Y, W, b, lamda):

    B = X.shape[1]

    P = evaluateClassifier(X, W, b)
    G = -(Y-P)

    grad_W = (1/B)*np.dot(G, X.T) + 2*lamda*W
    grad_b = np.mean(G, axis=-1, keepdims=True)

    return [grad_W, grad_b]


@jit(target="cuda")
def miniBatchGD(X, Y, GDparams, W, b, lamda):

    [grad_W, grad_b] = ComputeGradsNum(X, Y, W, b, lamda, 0.000001)

    W_t = W - GDparams['eta']*grad_W
    b_t = b - GDparams['eta']*grad_b

    return W_t, b_t


if __name__ == "__main__":

    # Take care with transpose or not
    # Take care with Y that goes from 0,...,9 and not 1,...,10
    training = LoadBatch('cifar-10-batches-py/data_batch_1')
    trainX = normalise(training[b'data'].T.astype(int))
    trainy = [int(i) for i in training[b'labels']]
    trainY = (np.eye(10)[trainy]).T

    validation = LoadBatch('cifar-10-batches-py/data_batch_2')
    validX = normalise(validation[b'data'].T.astype(int))
    validy = [int(i) for i in validation[b'labels']]
    validY = (np.eye(10)[validy]).T

    testing = LoadBatch('cifar-10-batches-py/test_batch')
    testX = normalise(testing[b'data'].T.astype(int))
    testy = [int(i) for i in testing[b'labels']]
    testY = (np.eye(10)[testy]).T

    # Dimension parameters
    d = trainX.shape[0]
    n = trainX.shape[1]
    K = len(np.unique(trainy))

    # Initialise W (Kxd) and b (Kx1)
    mu = 0
    sigma = 0.01
    W = np.random.normal(mu, sigma, size=(K, d))
    b = np.random.normal(mu, sigma, size=(K, 1))

    # Control parameters. GDparams
    n_batch = 100
    eta = 0.001
    n_epochs = 40
    GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs}

    lamda = 0
    h = 0.000001

    acc = []
    loss = []
    cost = []

    for n in range(n_epochs):
        for j in range(n_batch):
            print(j)
            N = int(trainX.shape[1] / n_batch)
            j_start = j*N
            j_end = (j+1)*N

            Xbatch = trainX[:, j_start:j_end]
            Ybatch = trainY[:, j_start:j_end]
            ybatch = trainy[j_start:j_end]

            W, b = miniBatchGD(Xbatch, Ybatch, GDparams, W, b, lamda)

            accuracy = ComputeAccuracy(Xbatch, ybatch, W, b)
            acc.append(accuracy)

            '''''
            print('\n\n loosing \n\n')
            loosing = crossentrpoyLoss(Xbatch, Ybatch, W, b)
            print(loosing)
            print('-------------')

            loss.append(loosing)
            '''''
        coste = 0
        coste = ComputeCost(Xbatch, Ybatch, W, b, lamda)
        cost.append(coste)
        print(cost)


        ''''
        loosing = 0
        perder = []

        for i in range(100):
            #print(i)
            loosing += crossentrpoyLoss(Xbatch[:, i], Ybatch[:, i], W, b)
            perder.append(crossentrpoyLoss(Xbatch[:, i], Ybatch[:, i], W, b))
            

        loss.append(np.argmax(perder))
        '''''

    #print(len(cost))
    print(cost)
    #print(acc)



    '''''
    X = trainX[:, 0]
    Y = trainY[:, 0]
    W = W[:, :]
    '''''

    # montage(W)

    '''''
    [ngrad_W, ngrad_b] = ComputeGradsNumSlow(X.reshape(len(X), 1), Y.reshape(len(Y), 1), W, b, lamda, h)
    [mgrad_W, mgrad_b] = ComputeGradients(X.reshape(len(X), 1), Y.reshape(len(Y), 1), W, b, lamda)
    check = np.sum(ngrad_b)-np.sum(mgrad_b)
    print(check)
    print(ngrad_W.shape)
    print(mgrad_W.shape)
    print(ngrad_b.shape)
    print(mgrad_b.shape)
    print(ngrad_W-mgrad_W)
    '''''

    # P = evaluateClassifier(trainX[:, :100], W, b)
    # print(trainX.shape[1])