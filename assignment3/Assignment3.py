from functions import *

import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.sparse
from math import floor


class DatasetManager:
    def __init__(self, recompute, filename, validation_filename):
        self.recompute = recompute
        self.dataset = self.ReadFile(filename, validation_filename)

        if recompute:
            self.oneHotLabels_train, self.oneHotNames_train, self.oneHotLabels_valid, self.oneHotNames_valid, self.\
                balance_train, self.balance_valid = self.RecomputeEncodedData(self.dataset)
        else:
            self.oneHotLabels_train, self.oneHotNames_train, self.oneHotLabels_valid, self.oneHotNames_valid, self.balance_train, self.balance_valid =\
                self.ReadEncodedData()

        self.names_train = self.dataset['names_train']
        self.labels_train = self.dataset['labels_train']
        self.names_validation = self.dataset['names_validation']
        self.labels_validation = self.dataset['labels_validation']
        self.alphabet = self.dataset['alphabet']
        self.d = self.dataset['d']
        self.K = self.dataset['K']
        self.n_len = self.dataset['n_len']

    def ReadFile(self, filename, validation_filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        with open(validation_filename, 'r') as f:
            lines_validation = f.readlines()

        validation_indexes = lines_validation[0][:-1].split(' ')
        validation_indexes = list(map(int, validation_indexes))

        dataset = {}
        names = []
        labels = []
        dataset["names_train"] = []
        dataset["labels_train"] = []
        dataset["names_validation"] = []
        dataset["labels_validation"] = []

        all_names = ""

        index = 0

        for line in lines:
            temp = line.replace(',', '').lower().split(' ')
            name = ""
            for i in range(len(temp) - 1):
                if i != 0:
                    name += ' '
                name += temp[i]
                all_names += temp[i]
            temp = temp[-1].replace('\n', '')

            names.append(name)
            labels.append(int(temp))

            if (index + 1) in validation_indexes:
                dataset["names_validation"].append(name)
                dataset["labels_validation"].append(int(temp) - 1)
            else:
                dataset["names_train"].append(name)
                dataset["labels_train"].append(int(temp) - 1)

            index += 1

        dataset["alphabet"] = 'abcdefghijklmnopqrstuvwxyz\' '
        dataset["d"] = len(dataset["alphabet"])
        dataset["K"] = len(list(set(labels)))
        dataset["n_len"] = len(max(names, key=len))

        dataset["labels_validation"] = np.array(dataset["labels_validation"])
        dataset["labels_train"] = np.array(dataset["labels_train"])

        return dataset

    def RecomputeEncodedData(self, dataset):

        oneHotLabels_train, balance_train = self.OneHotLabels(dataset['labels_train'], dataset['K'])
        oneHotNames_train = self.OneHotNames(dataset['names_train'], dataset['d'], dataset['n_len'],
                                             dataset['alphabet'])
        oneHotLabels_valid, balance_valid = self.OneHotLabels(dataset['labels_validation'], dataset['K'])
        oneHotNames_valid = self.OneHotNames(dataset['names_validation'], dataset['d'], dataset['n_len'],
                                             dataset['alphabet'])

        with open('EncodedData/labels_train.txt', 'wb') as f:
            np.save(f, oneHotLabels_train)

        with open('EncodedData/names_train.txt', 'wb') as f:
            np.save(f, oneHotNames_train)

        with open('EncodedData/labels_valid.txt', 'wb') as f:
            np.save(f, oneHotLabels_valid)

        with open('EncodedData/names_valid.txt', 'wb') as f:
            np.save(f, oneHotNames_valid)

        with open('EncodedData/balance_train.txt', 'wb') as f:
            np.save(f, balance_train)

        with open('EncodedData/balance_valid.txt', 'wb') as f:
            np.save(f, balance_valid)

        return oneHotLabels_train, oneHotNames_train, oneHotLabels_valid, oneHotNames_valid, balance_train, balance_valid

    def OneHotLabels(self, labels, K):

        n_labels = len(labels)
        oneHotLabels = np.zeros((K, n_labels), dtype=int)
        balance = np.zeros(K, dtype=int)

        for i in range(n_labels):
            progress = (i / n_labels) * 100
            sys.stdout.write('\rEncoding labels: ' + str(round(progress, 1)) + '%')

            label = labels[i]
            balance[labels[i]] += 1
            oneHotLabels[:, i][label] += 1

        return oneHotLabels, balance

    def OneHotNames(self, names, d, n_len, alphabet):

        first = True
        count = 0
        print('\n')

        for name in names:
            progress = (count / len(names)) * 100
            sys.stdout.write('\rEncoding names: ' + str(round(progress, 1)) + '%')

            oneHotName = np.zeros((d, n_len), dtype=int)

            for i in range(len(name)):
                letter = name[i]
                letterEncoded = alphabet.find(letter)

                oneHotName[:, i][letterEncoded] += 1

            oneHotName = oneHotName.T.reshape(-1)

            if first:
                oneHotNames = oneHotName
                first = False
            else:
                oneHotNames = np.vstack([oneHotNames, [oneHotName]])

            count += 1

        return oneHotNames.T

    def ReadEncodedData(self):

        with open('EncodedData/labels_train.txt', 'rb') as f:
            oneHotLabels_train = np.load(f)

        with open('EncodedData/names_train.txt', 'rb') as f:
            oneHotNames_train = np.load(f)

        with open('EncodedData/labels_valid.txt', 'rb') as f:
            oneHotLabels_valid = np.load(f)

        with open('EncodedData/names_valid.txt', 'rb') as f:
            oneHotNames_valid = np.load(f)

        with open('EncodedData/balance_train.txt', 'rb') as f:
            balance_train = np.load(f)

        with open('EncodedData/balance_valid.txt', 'rb') as f:
            balance_valid = np.load(f)

        return oneHotLabels_train, oneHotNames_train, oneHotLabels_valid, oneHotNames_valid, balance_train, balance_valid


class ConvNet:
    def __init__(self, dataset):

        self.data = dataset

        # CNN hyper-parameters
        self.n1 = 20  # number of filters in layer 1
        self.n2 = 20  # number of filters in layer 2
        self.k1 = 5  # width of filters applied in layer 1
        self.k2 = 3  # width of filters applied in layer 2

        # training parameters
        self.eta = 0.01
        self.rho = 0.9

        # dimensional parameters
        self.n_len = self.data.n_len
        self.n_len1 = self.n_len - self.k1 + 1
        self.n_len2 = self.n_len1 - self.k2 + 1

        self.F, self.W = self.Initialise(self.data)
        self.bestF, self. bestW = self.Initialise(self.data)

        self.F1_momentum = np.zeros(self.F[0].shape)
        self.F2_momentum = np.zeros(self.F[1].shape)
        self.W_momentum = np.zeros(self.W.shape)

    def Initialise(self, dataset):

        F = []
        F1 = np.random.randn(dataset.d, self.k1, self.n1) * (2 / self.n1)
        F.append(F1)
        F2 = np.random.randn(self.n1, self.k2, self.n2) * (2 / self.n2)
        F.append(F2)
        W = np.random.randn(dataset.K, self.n_len2*self.n2) * (2 / self.n2)

        #print('El n_len2 vale ' + str(self.n_len2))

        return F, W

    def ReluActivation(self, s):

        s[s < 0] = 0

        return s

    def fConvolutionMatrices(self, F, n_len, n_len1):

        (d1, k1, nf1) = F[0].shape
        M_F1 = np.zeros(((n_len - k1 + 1) * nf1, n_len * d1))
        V_F1 = F[0].reshape((d1 * k1, nf1), order='F').T

        for i in range(n_len - k1 + 1):
            M_F1[i * nf1:(i + 1) * nf1, d1 * i:d1 * i + d1 * k1] = V_F1

        (d2, k2, nf2) = F[1].shape
        M_F2 = np.zeros(((n_len1 - k2 + 1) * nf2, n_len1 * d2))
        V_F2 = F[1].reshape((d2 * k2, nf2), order='F').T

        for j in range(n_len1 - k2 + 1):
            M_F2[j * nf2:(j + 1) * nf2, d2 * j:d2 * j + d2 * k2] = V_F2

        return M_F1, M_F2

    def xConvolutionMatrices(self, x_vec, d, k, nf):

        n_len = int(x_vec.shape[0] / d)

        MX = np.zeros(((n_len - k + 1) * nf, k*nf*d))
        VX = np.zeros((n_len - k + 1, k*d))

        x_vec = x_vec.reshape((d, n_len), order='F')

        for i in range(n_len - k + 1):
            VX[i, :] = (x_vec[:, i:i + k].reshape((k * d, 1), order='F')).T

        for i in range(n_len - k + 1):
            for j in range(nf):
                MX[i*nf + j : i*nf + j + 1, j*k*d : j*k*d + k*d] = VX[i, :]

        return MX

    def ComputeLoss(self, X_batch, Ys_batch, MF, W):

        p, _, _ = self.EvaluateClassifier(X_batch, MF, W)
        B = X_batch.shape[1]

        if Ys_batch.shape == (len(Ys_batch),):
            Ys_batch = Ys_batch.reshape(len(Ys_batch), 1)

        loss = (1 / B) * (-(np.sum(Ys_batch * np.log(p))))

        return loss

    def EvaluateClassifier(self, X, MF, W):
        from functions import softmax

        #X_width = len(X[0])
        #X_height = len(X)

        x1 = self.ReluActivation(np.dot(MF[0], X)) #.reshape((X_width*X_height, 1), order='F'))
        X1 = x1#.T

        #X1_width = len(X1[0])
        #X1_height = len(X1)

        x2 = self.ReluActivation(np.dot(MF[1], X1))#.reshape((X1_width*X1_height, 1), order='F'))
        X2 = x2#.T

        #X2_width = len(X2[0])
        #X2_height = len(X2)

        #s = W * X2.reshape((X2_width*X2_height, 1), order='F')
        s = np.dot(W, X2)
        p = softmax(s)

        return p, X1, X2

    def ComputeGradients(self, X, Y, W, MX, idx):
        grad_F1 = np.zeros(self.F[0].shape)
        grad_F2 = np.zeros(self.F[1].shape)

        MF = self.fConvolutionMatrices(self.F, self.n_len, self.n_len1)

        P, X1, X2 = self.EvaluateClassifier(X, MF, W)
        G = -(Y-P)

        grad_W = (np.dot(G, X2.T))/X2.shape[1]

        G = np.dot(W.T, G)
        G = np.multiply(G, X2 > 0)

        n = X1.shape[1]
        for j in range(n):
            gj = G[:, [j]]
            xj = X1[:, [j]]

            Mj = self.xConvolutionMatrices(xj, self.n1, self.k2, self.n2)
            v = np.dot(gj.T, Mj)
            grad_F2 += v.reshape(self.F[1].shape, order='F') / n

        G = np.dot(MF[1].T, G)
        G = np.multiply(G, X1 > 0)

        n = X.shape[1]
        for j in range(n):
            gj = G[:, [j]]
            xj = X[:, [j]]

            #Mj = self.xConvolutionMatrices(xj, self.data.d, self.k1, self.n1)
            Mj = np.asarray(MX[idx[j]].todense())
            v = np.dot(gj.T, Mj)
            grad_F1 += v.reshape(self.F[0].shape, order='F') / n

        '''''
        self.W_momentum = self.W_momentum * self.rho + self.eta * grad_W
        self.F2_momentum = self.F2_momentum * self.rho + self.eta * grad_F2
        self.F1_momentum = self.F1_momentum * self.rho + self.eta * grad_F1
        '''''

        return grad_W, grad_F1, grad_F2

    def ComputeAccuracy(self, X, y, W, F):

        dataNum = X.shape[1]
        bingo = 0

        MF = self.fConvolutionMatrices(F, self.n_len, self.n_len1)

        for i in range(X.shape[1]):
            P, _, _ = self.EvaluateClassifier(X[:, i], MF, W)
            k_star = np.argmax(P)

            if k_star == y[i]:
                bingo += 1

        acc = bingo / dataNum

        return acc * 100

    def MiniBatchGD(self, X, Y, W, MX, idx):

        grad_W, grad_F1, grad_F2 = self.ComputeGradients(X, Y, W, MX, idx)

        ''''
        W_t = self.W - self.eta * grad_W
        F1_t = self.F[0] - self.eta * grad_F1
        F2_t = self.F[1] - self.eta * grad_F2
        '''''

        self.W_momentum = self.W_momentum * self.rho + self.eta * grad_W
        self.F2_momentum = self.F2_momentum * self.rho + self.eta * grad_F2
        self.F1_momentum = self.F1_momentum * self.rho + self.eta * grad_F1

        F1_t = self.F[0] - self.F1_momentum
        F2_t = self.F[1] - self.F2_momentum
        W_t = self.W - self.W_momentum

        return W_t, F1_t, F2_t

    def Train(self, trainX, trainY, validX, validY, validy, compensate=True):

        N = trainX.shape[1]
        dotter = 1
        batch_size = 100
        n_epochs = 2000

        accuracy = 0
        bestAccuracy = 0
        totalTrainCost = []
        totalValidateCost = []

        MX = []
        for j in range(trainX.shape[1]):
            if dotter == 1:
                dots = '.'
                dotter += 1
            elif dotter == 2:
                dots = '..'
                dotter += 1
            elif dotter == 3:
                dots = '...'
                dotter += 1
            elif dotter == 4:
                dots = '....'
                dotter += 1
            elif dotter == 5:
                dots = '.....'
                dotter = 1
            sys.stdout.write('\rPrecomputing' + dots)

            MX.append(scipy.sparse.csr_matrix(self.xConvolutionMatrices(
                trainX[:, [j]], self.data.d, self.k1, self.n1)))

        MX = np.array(MX)

        print('\nDONE PRECOMPUTING!\n')
        min_class = min(self.data.balance_train)

        if compensate:
            n_batch = floor((min_class * len(self.data.balance_train)) / batch_size)
        else:
            #n_batch = floor(N / batch_size)
            n_batch = 7

        for n in range(n_epochs):
            m = 0
            sys.stdout.write('\rEpoch number ' + str(n+1) + ' of ' + str(n_epochs))

            if compensate:
                for x in self.data.balance_train:
                    choices = np.random.randint(m, m+x, size=min_class)
                    if m == 0:
                        idx = choices
                    else:
                        idx = np.append(idx, choices)

                    m += x

                np.random.shuffle(idx)

            for j in range(n_batch):
                j_start = j * batch_size
                j_end = (j + 1) * batch_size
                if j == n_batch:
                    if compensate:
                        j_end = len(idx)
                    else:
                        j_end = N

                if compensate:
                    idx2 = idx[j_start: j_end]
                else:
                    idx2 = np.arange(j_start, j_end)

                Xbatch = trainX[:, idx2]
                Ybatch = trainY[:, idx2]

                self.W, self.F[0], self.F[1] = self.MiniBatchGD(Xbatch, Ybatch, self.W, MX, idx2)

                self.F[0] -= self.F1_momentum
                self.F[1] -= self.F2_momentum
                self.W -= self.W_momentum

            if n % 100 == 0:
                MF_1, MF_2 = self.fConvolutionMatrices(self.F, self.n_len, self.n_len1)
                MF = [MF_1, MF_2]

                trainCost = self.ComputeLoss(trainX, trainY, MF, self.W)
                validateCost = self.ComputeLoss(validX, validY, MF, self.W)

                totalTrainCost.append(trainCost)
                totalValidateCost.append(validateCost)

                accuracy = self.ComputeAccuracy(validX, validy, self.W, self.F)
                print('\nThe accuracy is: ' + str(accuracy))
                print('The train cost is: ' + str(trainCost))
                print('The validation cost is: ' + str(validateCost) + '\n')

                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    self.bestF = self.F
                    self.bestW = self.W

        self.ConfusionMatrix(trainX, trainY, MF)

        return totalTrainCost, totalValidateCost, accuracy, bestAccuracy

    def ConfusionMatrix(self, X, Y, MF):
        P, _, _ = self.EvaluateClassifier(X, MF, self.W)
        P = np.argmax(P, axis=0)
        T = np.argmax(Y, axis=0)

        M = np.zeros((self.data.K, self.data.K), dtype=int)

        np.set_printoptions(linewidth=100)

        for i in range(len(P)):
            M[T[i]][P[i]] += 1

        print('\n')
        print(M)

    def Plotter(self, trainCost, validateCost):
        plt.figure()
        plt.plot(list(range(len(trainCost))), trainCost, label='training cost')
        plt.plot(list(range(len(validateCost))), validateCost, label='validation cost')
        plt.xlabel("update step")
        plt.ylabel("cost")
        plt.legend()
        plt.savefig('./Result_Pics/cost78.png')

    def Test(self, X):
        MF = self.fConvolutionMatrices(self.bestF, self.n_len, self.n_len1)
        labels = ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German", "Greek", "Irish", "Italian",
                  "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese"]
        for i in range(X.shape[1]):
            P, _, _ = self.EvaluateClassifier(X[:, [i]], MF, self.bestW)
            label = np.argmax(P)

            print(names[i] + " is a name from " + labels[label])

    def SaveParameters(self):

        with open('Result_Parameters/best_F.txt', 'wb') as f:
            np.save(f, self.bestF)

        with open('Result_Parameters/best_W.txt', 'wb') as f:
            np.save(f, self.bestW)

    def ReadParameters(self):

        with open('Result_Parameters/best_F.txt', 'rb') as f:
            self.F = np.load(f, allow_pickle=True)

        self.bestF = self.F

        with open('Result_Parameters/best_W.txt', 'rb') as f:
            self.W = np.load(f, allow_pickle=True)

        self.bestW = self.W


if __name__ == "__main__":

    recompute = False
    compensate = True
    train = True
    train_names = "./Datasets/ascii_names.txt"
    valid_names = "./Datasets/Validation_Inds.txt"

    dataset = DatasetManager(recompute, train_names, valid_names)

    convnet = ConvNet(dataset)

    ''''
    print(len(dataset.oneHotLabels_train[0]))
    print(len(dataset.oneHotLabels_valid[0]))
    print(len(dataset.oneHotNames_train[0]))
    print(len(dataset.oneHotNames_valid[0]))

    print(dataset.oneHotLabels_train)
    print('\n')
    print(dataset.oneHotLabels_valid)
    '''''

    if train:
        trainCost, validateCost, finalAccuracy, bestAccuracy = convnet.Train(dataset.oneHotNames_train,
                                                                                              dataset.oneHotLabels_train,
                                                                                              dataset.oneHotNames_valid,
                                                                                              dataset.oneHotLabels_valid,
                                                                                              dataset.labels_validation,
                                                                                              compensate)

        convnet.Plotter(trainCost, validateCost)
        convnet.SaveParameters()

        print('\nThe final accuracy is: ' + str(finalAccuracy))
        print('\nThe best accuracy is: ' + str(bestAccuracy))
    else:
        convnet.ReadParameters()

    #print('\nThe resulting confusion Matrix is:')
    #print(confusionMatrix)

    names = ["Ferrer", "O'Neill", "Merkel", "Sutton", "Caracciolo", "Kurtz", "Manzanares", "Gonzalez", "Fraile", "Coya"]
    oneHotFriendNames = dataset.OneHotNames(names, dataset.d, dataset.n_len, dataset.alphabet)
    convnet.Test(oneHotFriendNames)
