import numpy as np

from functions import *
import matplotlib.pyplot as plt
import math


def loadOneBatchForEach():

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

    return trainX, trainy, trainY, validX, validy, validY, testX, testy, testY


def loadAllBatches():

    training = LoadBatch('cifar-10-batches-py/data_batch_1')
    trainX_1 = normalise(training[b'data'].T.astype(int))
    trainy_1 = [int(i) for i in training[b'labels']]
    trainY_1 = (np.eye(10)[trainy_1]).T

    training = LoadBatch('cifar-10-batches-py/data_batch_2')
    trainX_2 = normalise(training[b'data'].T.astype(int))
    trainy_2 = [int(i) for i in training[b'labels']]
    trainY_2 = (np.eye(10)[trainy_2]).T

    training = LoadBatch('cifar-10-batches-py/data_batch_3')
    trainX_3 = normalise(training[b'data'].T.astype(int))
    trainy_3 = [int(i) for i in training[b'labels']]
    trainY_3 = (np.eye(10)[trainy_3]).T

    training = LoadBatch('cifar-10-batches-py/data_batch_4')
    trainX_4 = normalise(training[b'data'].T.astype(int))
    trainy_4 = [int(i) for i in training[b'labels']]
    trainY_4 = (np.eye(10)[trainy_4]).T

    training = LoadBatch('cifar-10-batches-py/data_batch_5')
    trainX_5 = normalise(training[b'data'].T.astype(int))
    trainy_5 = [int(i) for i in training[b'labels']]
    trainY_5 = (np.eye(10)[trainy_5]).T

    trainX = np.concatenate((trainX_1, trainX_2, trainX_3, trainX_4, trainX_5), axis=1)
    trainY = np.concatenate((trainY_1, trainY_2, trainY_3, trainY_4, trainY_5), axis=1)
    trainy = np.concatenate((trainy_1, trainy_2, trainy_3, trainy_4, trainy_5))

    validX = trainX[:, -1000:]
    validY = trainY[:, -1000:]
    validy = trainy[-1000:]

    trainX = trainX[:, :-1000]
    trainY = trainY[:, :-1000]
    trainy = trainy[:-1000]

    testing = LoadBatch('cifar-10-batches-py/test_batch')
    testX = normalise(testing[b'data'].T.astype(int))
    testy = [int(i) for i in testing[b'labels']]
    testY = (np.eye(10)[testy]).T

    return trainX, trainy, trainY, validX, validy, validY, testX, testy, testY


def normalise(matrix):

    matrixMean = np.mean(matrix, axis=1)
    stdMatrix = np.std(matrix, axis=1)
    repetitions = len(matrix[0])

    matrixMean = np.transpose([matrixMean]*repetitions)
    stdMatrix = np.transpose([stdMatrix]*repetitions)

    matrix = matrix - matrixMean
    matrix = matrix / stdMatrix

    return matrix


def initialise(d, m, K, mu=0):

    sigma1 = 1/math.sqrt(d)
    sigma2 = 1/math.sqrt(m)

    W1 = np.random.normal(mu, sigma1, size=(m, d))
    W2 = np.random.normal(mu, sigma2, size=(K, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((K, 1))

    W = [W1, W2]
    b = [b1, b2]

    return W, b


def reluActivation(s):

    s[s < 0] = 0

    return s


def evaluateClassifier(X, W, b):
    from functions import softmax
    # X (dx1)
    # W1 (mxd)
    # W2 (Kxm)
    # b1 (mx1)
    # b2 (Kx1)
    # p (Kx1)

    if X.shape == (len(X),):
        X = X.reshape(len(X), 1)

    s1 = np.dot(W[0], X) + b[0]
    h = reluActivation(s1)
    s = np.dot(W[1], h) + b[1]
    p = softmax(s)

    return p, h


def ComputeCost(X, Y, W, b, lamda):

    loss = crossentrpoyLoss(X, Y, W, b)
    reg_term = np.sum(W[0]**2) + np.sum(W[1]**2)

    J = loss + (lamda*reg_term)

    return J


def crossentrpoyLoss(X, Y, W, b):

    p, _ = evaluateClassifier(X, W, b)
    D = X.shape[1]

    if Y.shape == (len(Y),):
        Y = Y.reshape(len(Y), 1)

    loss = (1/D) * (-(np.sum(Y*np.log(p))))

    return loss


def ComputeAccuracy(X, y, W, b):

    dataNum = X.shape[1]
    bingo = 0

    for i in range(X.shape[1]):
        P, _ = evaluateClassifier(X[:, i], W, b)
        k_star = np.argmax(P)

        if k_star == y[i]:
            bingo += 1

    acc = bingo/dataNum

    return acc


def ComputeGradients(X, Y, W, b, lamda):

    B = X.shape[1]
    P, h = evaluateClassifier(X, W, b)
    G = -(Y-P)

    grad_W2 = (1/B)*np.dot(G, h.T) + 2*lamda*W[1]
    grad_b2 = np.reshape((1/B)*np.dot(G, np.ones(B)), (10, 1))

    # Back propagate the gradient through 2nd fully connected layer
    G = np.dot(W[1].T, G)
    G = np.multiply(G, h > 0)

    grad_W1 = (1/B)*np.dot(G, X.T) + 2*lamda*W[0]
    grad_b1 = np.reshape((1/B)*np.dot(G, np.ones(B)), (50, 1))

    grad_W = [grad_W1, grad_W2]
    grad_b = [grad_b1, grad_b2]

    return [grad_W, grad_b]


def miniBatchGD(X, Y, GDparams, W, b, lamda):

    [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lamda)

    W1_t = W[0] - GDparams['eta']*grad_W[0]
    W2_t = W[1] - GDparams['eta']*grad_W[1]
    b1_t = b[0] - GDparams['eta']*grad_b[0]
    b2_t = b[1] - GDparams['eta']*grad_b[1]

    W_t = [W1_t, W2_t]
    b_t = [b1_t, b2_t]

    return W_t, b_t


def saveData(totalCost, totalLoss, totalAcc, t):
    plt.figure(1)
    plt.plot(t, totalCost[0], label='training cost')
    plt.plot(t, totalCost[1], label='validation cost')
    plt.ylim([0, 4])
    plt.xlabel("update step")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig('Result_Pics/BNcost.png')

    plt.figure(2)
    plt.plot(t, totalLoss[0], label='training loss')
    plt.plot(t, totalLoss[1], label='validation loss')
    plt.ylim([0, 3.5])
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('Result_Pics/BNloss.png')

    plt.figure(3)
    plt.plot(t, totalAcc[0], label='training accuracy')
    plt.plot(t, totalAcc[1], label='validation accuracy')
    plt.ylim([0, 0.8])
    plt.xlabel("update step")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('Result_Pics/BNaccuracy.png')


if __name__ == "__main__":

    #trainX, trainy, trainY, validX, validy, validY, testX, testy, testY = loadOneBatchForEach()
    trainX, trainy, trainY, validX, validy, validY, testX, testy, testY = loadAllBatches()


    # Dimension parameters
    d = trainX.shape[0]
    n = trainX.shape[1]
    K = len(np.unique(trainy))

    # Initialise W (Kxd) and b (Kx1)
    mu = 0
    sigma = 0.01
    m = 50
    W, b = initialise(d, m, K, mu)

    # Control parameters. GDparams
    n_batch = 100
    l = 0
    eta_min = 0.00001
    eta_max = 0.1
    n_s = 2 * math.floor(n/n_batch)
    eta = eta_min
    n_epochs = 200
    GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs}

    lamda = 0.0010283387653801572
    h_param = 0.000001

    # lambda regularisation
    l_min = -3
    l_max = -1.5
    best_accuracy = 0
    best_lamda = 0

    # TRAIN THE MODEL
    t = -1
    total_t = 0
    accuracy = 0
    cycles = 3

    totalTrainCost = []
    totalValidateCost = []

    totalTrainLoss = []
    totalValidateLoss = []

    totalTrainAcc = []
    totalValidateAcc = []

    totalT = []
    etaEv = []

    N = int(trainX.shape[1] / n_batch)
    '''''
    for i in range(20):

        t = -1
        total_t = 0

        print('******************')
        print('Network candidate' + str(i))
        print('******************')

        l_epoch = l_min + (l_max-l_min)*np.random.rand(1, 1)[0][0]
        lamda = 10**l_epoch

    '''''

    for n in range(n_epochs):

        trainCost = ComputeCost(trainX, trainY, W, b, lamda)
        validateCost = ComputeCost(validX, validY, W, b, lamda)
        trainLoss = crossentrpoyLoss(trainX, trainY, W, b)
        validateLoss = crossentrpoyLoss(validX, validY, W, b)
        trainAcc = ComputeAccuracy(trainX, trainy, W, b)
        validateAcc = ComputeAccuracy(validX, validy, W, b)

        totalTrainCost.append(trainCost)
        totalValidateCost.append(validateCost)
        totalTrainLoss.append(trainLoss)
        totalValidateLoss.append(validateLoss)
        totalTrainAcc.append(trainAcc)
        totalValidateAcc.append(validateAcc)
        totalT.append(total_t)

        for j in range(n_batch):
            j_start = j*N
            j_end = (j+1)*N

            Xbatch = trainX[:, j_start:j_end]
            Ybatch = trainY[:, j_start:j_end]
            ybatch = trainy[j_start:j_end]

            t += 1
            if (t >= 2*l*n_s) and (t <= (2*l+1)*n_s):
                GDparams['eta'] = eta_min + ((t-2*l*n_s)/n_s)*(eta_max-eta_min)

            if (t >= (2*l+1)*n_s) and (t <= 2*(l+1)*n_s):
                GDparams['eta'] = eta_max - ((t-(2*l+1)*n_s)/n_s)*(eta_max-eta_min)

                if t == 2*(l+1)*n_s:
                    t = -1

            etaEv.append(GDparams['eta'])

            total_t += 1
            #print('Update step: ' + str(total_t))

            if total_t >= cycles * 2*n_s:
                break

            W, b = miniBatchGD(Xbatch, Ybatch, GDparams, W, b, lamda)

        print('Epoch ' + str(n) + ' of ' + str(n_epochs))

        if total_t >= cycles * 2*n_s:
            break
        ''''
        accuracy_epoch = ComputeAccuracy(validX, validy, W, b)

        if accuracy_epoch > best_accuracy:
            best_accuracy = accuracy_epoch
            best_lamda = lamda
        '''

        #print('\n')
        ''''
        file = open("Result_Pics/hyper_parameters_fine.txt", "a")
        file.write('\nCandidate number ' + str(i) + ' has an accuracy of ' + str(accuracy_epoch) + ' for a lambda of ' + str(lamda))
        file.close()
        '''''


    #print('The best accuracy reached is ' + str(best_accuracy) + ' for a lambda of ' + str(best_lamda))
    print('\n\nThe reached accuracy is: ')
    print(ComputeAccuracy(testX, testy, W, b))

    totalCost = [totalTrainCost, totalValidateCost]
    totalLoss = [totalTrainLoss, totalValidateLoss]
    totalAcc = [totalTrainAcc, totalValidateAcc]

    # print(ComputeAccuracy(testX, testy, W, b))

    saveData(totalCost, totalLoss, totalAcc, totalT)
    '''''
    plt.close("all")
    plt.figure(4)
    plt.plot(list(range(total_t)), etaEv)
    plt.xlabel("update step")
    plt.ylabel("eta")
    plt.show()
    '''''
