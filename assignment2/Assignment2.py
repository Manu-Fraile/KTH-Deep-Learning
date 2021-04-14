from functions import *
import matplotlib.pyplot as plt
import math


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
    b1 = np.random.normal(mu, sigma1, size=(m, 1))
    b2 = np.random.normal(mu, sigma2, size=(K, 1))

    W = [W1, W2]
    b = [b1, b2]

    return W, b


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

    print(len(W))
    print('W1: ' + str(len(W[0])) + 'x' + str(len((W[0])[0])))
    print('W2: ' + str(len(W[1])) + 'x' + str(len((W[1])[0])))
    print('b1: ' + str(len(b[0])) + 'x' + str(len((b[0])[0])))
    print('b2: ' + str(len(b[1])) + 'x' + str(len((b[1])[0])))
    print('X: ' + str(X.shape))
    #print(W)
    #print('\n')
    #print((W[0])[0])

    s1 = np.dot(W[0], X) + b[0]
    print('\n')
    print('s1: ' + str(len(s1)) + 'x' + str(len(s1[0])))
    h = max(0, s1.all())
    s = np.dot(W[1], h) + b[1]
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

    if Y.shape == (len(Y),):
        Y = Y.reshape(len(Y), 1)

    return (-1)*(np.sum(Y*np.log(p)))


def ComputeAccuracy(X, y, W, b):

    dataNum = X.shape[1]
    bingo = 0

    for i in range(X.shape[1]):
        P = evaluateClassifier(X[:, i], W, b)
        k_star = np.argmax(P)

        if k_star == y[i]:
            bingo += 1

    acc = bingo/dataNum

    return acc


def ComputeGradients(X, Y, W, b, lamda):

    B = X.shape[1]

    print('ComputeGradients: ' + str(X.shape))


    P = evaluateClassifier(X, W, b)
    G = -(Y-P)

    grad_W = (1/B)*np.dot(G, X.T) + 2*lamda*W
    grad_b = np.mean(G, axis=-1, keepdims=True)

    return [grad_W, grad_b]


def miniBatchGD(X, Y, GDparams, W, b, lamda):

    print('minibatchGD: ' + str(X.shape))

    [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lamda)

    W_t = W - GDparams['eta']*grad_W
    b_t = b - GDparams['eta']*grad_b

    return W_t, b_t


def saveData(trainCost, validateCost, n_epochs, accuracy):
    plt.plot(list(range(n_epochs)), trainCost, label='train loss')
    plt.plot(list(range(n_epochs)), validateCost, label='validation loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('Result_Pics/cost4.png')

    file = open("Result_Pics/Accuracy.txt", "a")
    file.write('\nAccuracy 4: ' + str(accuracy))
    file.close()


if __name__ == "__main__":

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
    m = 50
    W, b = initialise(d, m, K, mu)

    # Control parameters. GDparams
    n_batch = 100
    eta = 0.001
    n_epochs = 40
    GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': n_epochs}

    lamda = 1
    h_param = 0.000001

    # TRAIN THE MODEL
    accuracy = 0
    totalTrainCost = []
    totalValidateCost = []

    for n in range(n_epochs):
        for j in range(n_batch):
            N = int(trainX.shape[1] / n_batch)
            j_start = j*N
            j_end = (j+1)*N

            Xbatch = trainX[:, j_start:j_end]
            Ybatch = trainY[:, j_start:j_end]
            ybatch = trainy[j_start:j_end]

            W, b = miniBatchGD(Xbatch, Ybatch, GDparams, W, b, lamda)

        trainCost = ComputeCost(trainX, trainY, W, b, lamda)
        validateCost = ComputeCost(validX, validY, W, b, lamda)

        totalTrainCost.append(trainCost)
        totalValidateCost.append(validateCost)

    accuracy = ComputeAccuracy(testX, testy, W, b)

    saveData(totalTrainCost, totalValidateCost, n_epochs, accuracy)
    montage(W)
