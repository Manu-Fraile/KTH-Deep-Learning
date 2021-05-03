from functions import *
#from ExtractNames import *

import numpy as np
import sys
import matplotlib.pyplot as plt
import math


class DatasetManager:
    def __init__(self, recompute, filename, validation_filename):
        self.recompute = recompute
        self.dataset = self.ReadFile(filename, validation_filename)

        if recompute:
            self.oneHotLabels, self.oneHotNames = self.RecomputeEncodedData()
        else:
            self.oneHotLabels, self.oneHotNames = self.ReadEncodedData()

        self.names_train = self.dataset['names_train']
        self.labels_train = self.dataset['labels_train']
        self.names_validation = self.dataset['names_validation']
        self.labels_validation = self.dataset['labels_validation']
        self.alphabet = self.dataset['alphabet']
        self.d = self.dataset['d']
        self.K = self.dataset['K']
        self.n_letters = self.dataset['n_len']

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

        oneHotLabels = self.OneHotLabels(dataset['labels_train'], dataset['K'])
        oneHotNames = self.OneHotNames(dataset['names_train'], dataset['d'], dataset['n_letters'], dataset['alphabet'])

        with open('EncodedData/labels.txt', 'wb') as f:
            np.save(f, oneHotLabels)

        with open('EncodedData/names.txt', 'wb') as f:
            np.save(f, oneHotNames)

        return oneHotLabels, oneHotNames

    def OneHotLabels(self, labels, K):

        n_labels = len(labels)
        oneHotLabels = np.zeros((K, n_labels), dtype=int)

        for i in range(n_labels):
            progress = (i / n_labels) * 100
            sys.stdout.write('\rEncoding labels: ' + str(round(progress, 1)) + '%')

            label = labels[i]
            oneHotLabels[:, i][label] += 1

        return oneHotLabels

    def OneHotNames(self, names, d, n_letters, alphabet):

        first = True
        count = 0
        print('\n')

        for name in names:
            progress = (count / len(names)) * 100
            sys.stdout.write('\rEncoding names: ' + str(round(progress, 1)) + '%')

            oneHotName = np.zeros((d, n_letters), dtype=int)

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

        with open('EncodedData/labels.txt', 'rb') as f:
            oneHotLabels = np.load(f)

        with open('EncodedData/names.txt', 'rb') as f:
            oneHotNames = np.load(f)

        return oneHotLabels, oneHotNames



class ConvNet:
    def __init__(self, F, W):
        self.F = F
        self.W = W


def ReluActivation(s):

    s[s < 0] = 0

    return s


def fConvolutionMatrices(F, n_letters):

    (d1, k1, nf1) = F[0].shape
    M_F1 = np.zeros(((n_letters - k1 + 1) * nf1, n_letters * d1))
    V_F1 = F[0].reshape((d1 * k1, nf1), order='F').T

    for i in range(n_letters - k1 + 1):
        M_F1[i * nf1:(i + 1) * nf1, d1 * i:d1 * i + d1 * k1] = V_F1

    (d2, k2, nf2) = F[1].shape
    M_F2 = np.zeros(((n_letters - k2 + 1) * nf2, n_letters * d2))
    V_F2 = F[1].reshape((d2 * k2, nf2), order='F').T

    for i in range(n_letters - k2 + 1):
        M_F2[i * nf2:(i + 1) * nf2, d2 * i:d2 * i + d2 * k2] = V_F2

    return M_F1, M_F2


def xConvolutionMatrices(x_vec, d, k, nf):

    n_letters = int(x_vec.shape[0] / d)

    MX = np.zeros(((n_letters - k + 1) * nf, k * nf * d))
    VX = np.zeros((n_letters - k + 1, k * d))

    x_vec = x_vec.reshape((d, n_letters), order='F')

    for i in range(n_letters - k + 1):
        VX[i, :] = (x_vec[:, i:i + k].reshape((k * d, 1), order='F')).T

    for i in range(n_letters - k + 1):
        for j in range(nf):
            MX[i * nf + j:i * nf + j + 1, j * k * d:j * k * d + k * d] = VX[i, :]

    return MX


def ComputeLoss(X_batch, Ys_batch, MF, W):

    p, _, _ = EvaluateClassifier(X_batch, MF, W)
    B = X_batch.shape[1]

    if Ys_batch.shape == (len(Ys_batch),):
        Ys_batch = Ys_batch.reshape(len(Ys_batch), 1)

    loss = (1 / B) * (-(np.sum(Ys_batch * np.log(p))))

    return loss


def EvaluateClassifier(X, MF, W):
    from functions import softmax

    X_width = len(X[0])
    X_height = len(X)

    x1 = ReluActivation(MF[0] * X.reshape((X_width*X_height, 1), order='F'))
    X1 = x1.T

    X1_width = len(X1[0])
    X1_height = len(X1)

    x2 = ReluActivation(MF[1] * X1.reshape((X1_width*X1_height, 1), order='F'))
    X2 = x2.T

    X2_width = len(X2[0])
    X2_height = len(X2)

    s = W * X2.reshape((X2_width*X2_height, 1), order='F')
    p = softmax(s)

    return p, X1, X2


def ComputeGradients(X, Y, W, b, lamda):
    grad_F1 = np.zeros((self.F1.shape))
    grad_F2 = np.zeros((self.F2.shape))
    grad_W = np.zeros((W.shape))

    D = X.shape[1]
    P, X1, X2 = EvaluateClassifier(X, MF, W)
    G = -(Y-P)

    grad_W = (G*X2.T)/X2.shape[1]

    G = np.dot(W[1].T, G)
    G = np.multiply(G, X2 > 0)

    n = X1.shape[1]
    for j in range(n):
        xj = X1[:, [j]]
        gj = G[:, [j]]

        Mj = self.MXMatrix(
              xj, self.dimensions[0], self.dimensions[3], self.dimensions[2])
        v = np.dot(gj.T, Mj)
        grad_F2 += v.reshape(self.F2.shape, order='F') / n


        a = gj.shape[0]
        gj = gj.reshape((int(a / self.dimensions[2]), self.dimensions[2]))

        v2 = np.dot(MjGen.T, gj)

        gradF2 += v2.reshape(self.F2.shape, order='F') / n



    grad_W2 = (1/D)*np.dot(G, h.T) + 2*lamda*W[1]
    grad_b2 = np.reshape((1/B)*np.dot(G, np.ones(B)), (10, 1))

    # Back propagate the gradient through 2nd fully connected layer
    G = np.dot(W[1].T, G)
    G = np.multiply(G, h > 0)

    grad_W1 = (1/B)*np.dot(G, X.T) + 2*lamda*W[0]
    grad_b1 = np.reshape((1/B)*np.dot(G, np.ones(B)), (50, 1))

    grad_W = [grad_W1, grad_W2]
    grad_b = [grad_b1, grad_b2]

    return [grad_W, grad_b]


if __name__ == "__main__":

    recompute = False
    train_names = "./Datasets/ascii_names.txt"
    valid_names = "./Datasets/Validation_Inds.txt"

    dataset = DatasetManager(recompute, train_names, valid_names)

    print(dataset.n_letters)

    '''''
    filename = "./Datasets/ascii_names.txt"
    validation_filename = "./Datasets/Validation_Inds.txt"
    dataset = read_file(filename, validation_filename)

    if recompute:
        oneHotLabels, oneHotNames = RecomputeEncodedData(dataset)
    else:
        oneHotLabels, oneHotNames = ReadEncodedData()
    '''''

    # CNN hyper-parameters
    n1 = 5     # number of filters in layer 1
    n2 = 5     # number of filters in layer 2
    k1 = 5      # width of filters applied in layer 1
    k2 = 5      # width of filters applied in layer 2

    # training parameters
    eta = 0.001
    tho = 0.9

    F = []
    F1 = np.random.randn(dataset.d, k1, n1) * (2 / n1)
    F.append(F1)
    F2 = np.random.randn(n1, k2, n2) * (2 / n2)
    F.append(F2)
    W = np.random.randn(dataset.K, n2) * (2 / n2)

    convnet = ConvNet(F, W)

    '''''
    print(oneHotNames.shape)
    print(len(oneHotNames))
    print(len(oneHotNames[0]))
    '''''
