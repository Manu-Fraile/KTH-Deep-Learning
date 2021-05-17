import random

import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from math import floor


class DatasetManager:
    def __init__(self, datafile):
        self.book, self.uchars, self.d = self.ReadFile(datafile)

    def ReadFile(self, datafile):
        book_data = ''
        with open(datafile, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            book_data += line

        book_char = []

        for i in range(len(book_data)):
            if not (book_data[i] in book_char):
                book_char.append(book_data[i])
        # print(len(book_char))

        return book_data, book_char, len(book_char)

    def char_to_ind(self, char):
        ind = np.zeros((self.d, 1), dtype=int)
        ind[self.uchars.index(char)] = 1
        return ind.T

    def ind_to_char(self, ind):
        return self.uchars[np.argmax(ind)]


class RNN:
    def __init__(self, dataset, sig=0.01):
        self.data = dataset

        # Gradient parameters
        self.m = 100
        self.theta = 0

        # Initialisation parameters
        self.seq_length = 25
        self.h = np.zeros((self.m, self.seq_length))
        self.h_minus1 = np.zeros((self.m, 1))
        self.h_plus1 = np.zeros((self.m, 1))
        '''
        self.X_chars = '.'
        self.Y_chars = '.'
        self.X = np.zeros((self.data.d, self.seq_length))
        self.Y = np.zeros((self.data.d, self.seq_length))
        '''''

        # Optimisation parameters
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.data.d, 1))
        self.W = np.random.randn(self.m, self.m) * sig
        self.U = np.random.randn(self.m, self.data.d) * sig
        self.V = np.random.randn(self.data.d, self.m) * sig

        # Training parameters
        self.eta = 0.1
        self.epsilon = 1e-8

        # Loss parameters
        #self.loss = 0
        #self.smooth_loss = 0

        self.m_b = np.zeros((self.m, 1))
        self.m_c = np.zeros((self.data.d, 1))
        self.m_U = np.zeros((self.m, self.data.d))
        self.m_W = np.zeros((self.m, self.m))
        self.m_V = np.zeros((self.data.d, self.m))

    def OneHotEncoder(self, text):
        encoded = []
        first = True

        for char in text:
            charEncoded = self.data.char_to_ind(char)

            if first:
                encoded = charEncoded
                first = False
            else:
                encoded = np.vstack((encoded, charEncoded))

        return encoded.T

    def EvaluateClassifier(self, X):
        from functions import softmax

        #print('Pre-Evaluate:' + str(self.h.shape))
        #print('The X: ' + str(X.shape)) # (80, 25)

        ''''
        print('\nEl evaluate Classifier')
        print('V: ' + str(self.V.shape))
        print('h: ' + str(self.h.shape))
        print('c: ' + str(self.c.shape))
        print('b: ' + str(self.b.shape))
        '''''

        p = np.zeros((self.data.d, self.seq_length))
        #print(self.h.shape)

        for t in range(self.seq_length):
            a = np.dot(self.W, self.h[:, [t-1]]) + np.dot(self.U, X[:, [t]]) + self.b
            self.h[:, [t]] = np.tanh(a)
            o = np.dot(self.V, self.h[:, [t]]) + self.c
            pt = softmax(o)

            p[:, [t]] = pt

        '''''
        a = np.dot(self.W, self.h) + np.dot(self.U, X) + np.repeat(self.b, self.seq_length, axis=1)
        #print('a: ' + str(a.shape))

        self.h = np.tanh(a)
        #print('h post: ' + str(self.h.shape))

        o = np.dot(self.V, self.h) + np.repeat(self.c, self.seq_length, axis=1)
        #print('o: ' + str(o.shape))

        p = softmax(o)
        #print('p: ' + str(p.shape))
        '''''

        ''''
        self.b = self.b.reshape(100,1)
        print(self.b.shape)
        print(np.repeat(self.b, 3, axis=1))
        '''''

        #print('\nTHE P: ' + str(p))

        #print('Post-Evaluate:' + str(self.h.shape))

        #print('\nTHE P IS: ' + str(p.shape))
        #print(p)

        return p

    def ComputeLoss(self, X, Y):

        P = self.EvaluateClassifier(X)
        loss = 0
        smooth_loss = 0

        for t in range(self.seq_length):
            loss += -np.log(np.dot(Y[:, t].T, P[:, t]))

        #loss = -(np.sum(np.log(np.dot(Y.T, P))))
        smooth_loss = 0.999*smooth_loss + 0.001*loss

        return loss, smooth_loss

    def ComputeGradients(self, X, Y):

        #print('El size de X: ' + str(self.X.shape))

        P = self.EvaluateClassifier(X)

        grad_W = np.zeros_like(self.W)
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_b = np.zeros_like(self.b)
        grad_c = np.zeros_like(self.c)

        #print(self.h.shape)

        grad_o = -(Y - P)
        #grad_o = grad_o.T

        #print('\nLa shape de o: ') #Esto esta bien: (80, 25)
        #print(grad_o.shape)

        grad_h = np.zeros_like(self.h)
        #print(grad_h.shape)
        #print(grad_h[:, 20].shape)

        for t in reversed(range(self.seq_length-1)):
            diag_tplus1 = np.diagflat(1-self.h[:, t+1]**2)

            grad_h_t = self.W.T.dot(grad_h[:, t+1]).dot(diag_tplus1) + np.dot(self.V.T, grad_o[:, t])
            grad_h[:, t] = grad_h_t

        #grad_h = np.fliplr(grad_h)
        #print('El gradiente de h')
        #print(grad_h.shape)
        #print('\n\n')

        #print(grad_h)

        #print(diag_t.shape)

        for t in range(self.seq_length):
            diagonal = np.diagflat(1-self.h[:, t]**2)
            #print('\nDale otro niño')
            #print(diagonal.shape)
            #print(grad_h[:, t].shape)
            #print(self.h[:, t-1].reshape(100, 1).T.shape)

            #print('El data d: ' + str(self.data.d))

            grad_W += diagonal.dot(grad_h[:, [t]]).dot(self.h[:, [t-1]].T)
            grad_U += diagonal.dot(grad_h[:, [t]]).dot(X[:, [t]].T)
            grad_b += diagonal.dot(grad_h[:, [t]])
            #grad_V += np.dot(grad_o[:, t], self.h.T)
            #print('La shape de W:' + str(grad_W.shape))

        #print('El gradiente de W:' + str(grad_W.shape))

        #print(grad_W)

        #grad_W = np.sum(np.diagflat(1-self.h**2) * grad_h * self.h_plus1.T, axis=1)

        #grad_U = np.sum(np.diagflat(1-self.h**2) * grad_h * X.T, axis=1)

        #print('\nLa shape de las cosas de V:')
        #print(grad_o.shape)
        #print(self.h.T.shape)

        grad_V = np.dot(grad_o, self.h.T)
        grad_V = np.zeros_like(self.V)

        #print('\nLa shape despues del sum: ')
        #print(grad_V.shape)

        #grad_b = np.sum(np.diagflat(1-self.h**2) * grad_h, axis=1)

        grad_c = np.sum(grad_o, axis=-1, keepdims=True)

        grad_W = np.clip(grad_W, -5, 5)
        grad_U = np.clip(grad_U, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)

        return grad_W, grad_U, grad_V, grad_b, grad_c

    def VanillaSGD(self, X, Y):
        grad_W, grad_U, grad_V, grad_b, grad_c = self.ComputeGradients(X, Y)

        #print('\nLa shape de las Vs: ')
        #print(grad_V.shape)
        #print(self.V.shape)

        ''''
        print('\nThe shape of W ' + str(self.W.shape) + ' and of grad_W ' + str(grad_W.shape))
        print('The shape of U ' + str(self.U.shape) + ' and of grad_U ' + str(grad_U.shape))
        print('The shape of V ' + str(self.V.shape) + ' and of grad_V ' + str(grad_V.shape))
        print('The shape of b ' + str(self.b.shape) + ' and of grad_b ' + str(grad_b.shape))
        print('The shape of c ' + str(self.c.shape) + ' and of grad_c ' + str(grad_c.shape))
        '''''

        self.m_b += np.multiply(grad_b, grad_b)
        self.m_c += np.multiply(grad_c, grad_c)
        self.m_U += np.multiply(grad_U, grad_U)
        self.m_W += np.multiply(grad_W, grad_W)
        self.m_V += np.multiply(grad_V, grad_V)

        self.b -= np.multiply(self.eta / np.sqrt(self.m_b + self.epsilon), grad_b)
        self.c -= np.multiply(self.eta / np.sqrt(self.m_c + self.epsilon), grad_c)
        self.U -= np.multiply(self.eta / np.sqrt(self.m_U + self.epsilon), grad_U)
        self.W -= np.multiply(self.eta / np.sqrt(self.m_W + self.epsilon), grad_W)
        self.V -= np.multiply(self.eta / np.sqrt(self.m_V + self.epsilon), grad_V)

        ''''
        self.W = self.W - self.eta*grad_W
        self.U = self.U - self.eta*grad_U
        self.V = self.V - self.eta*grad_V
        self.b = self.b - self.eta*grad_b
        self.c = self.c - self.eta*grad_c
        '''''

    def Train(self, n_epochs):

        #X = self.OneHotEncoder(self.X_chars)
        #N = X.shape[1]
        batch_size = 100

        total_loss = []

        for n in range(n_epochs):
            sys.stdout.write('\rEpoch number ' + str(n+1) + ' of ' + str(n_epochs))
            hprev = np.zeros_like(self.h)

            for i in range(1, 20000):
                hprev = self.h
                sys.stdout.write('\rIteration number ' + str(i + 1) + ' of ' + str(44302))

                for e in range(self.seq_length):
                    init = e + i*self.seq_length
                    end = e + self.seq_length*(i+1)
                    #print('\n Start:' + str(init))
                    #print('\n End:' + str(end))
                    X_chars = self.data.book[init:end]
                    Y_chars = self.data.book[init+1:end+1]

                    X = self.OneHotEncoder(X_chars)
                    #print('\nLa shape de X: ' + str(X.shape))
                    Y = self.OneHotEncoder(Y_chars)
                    #print('\nLa shape de Y: ' + str(Y.shape))


                    self.VanillaSGD(X, Y)

                if i % 1000 == 0:
                    loss, smooth_loss = self.ComputeLoss(X, Y)
                    #print('\nTHE LOSS: ' + str(loss))
                    #print('THE SMOOTH_LOSS: ' + str(smooth_loss) + '\n')
                    #print("ite:", i, "smooth_loss:", smooth_loss)
                    total_loss.append(smooth_loss)

                if i % 1000 == 0:
                    Y_temp = self.SynthText(X, hprev, 200)
                    string = ""
                    for j in range(Y_temp.shape[1]):
                        string += self.data.ind_to_char(Y_temp[:, [j]])

                    print(string)

        Y_temp = self.SynthText(self.data.char_to_ind("H", self.data.uchars).T, self.h, 1000)
        string = ""
        for i in range(Y_temp.shape[1]):
            string += self.data.ind_to_char(Y_temp[:, [i]], self.data.uchars)
        print(string)

        self.SaveParameters(total_loss)

    def SynthText(self, X, hprev, a):

        Y = np.zeros((self.data.d, a))
        x = X
        self.h = hprev

        for i in range(a):
            p = self.EvaluateClassifier(X)
            label = np.random.choice(self.data.d, p=p[:, 0])

            Y[label][i] = 1
            x = np.zeros(x.shape)
            x[label] = 1

        return Y

    def SaveParameters(self, loss):

        with open('Result_Parameters/total_loss.txt', 'wb') as f:
            np.save(f, loss)

    def ReadParameters(self):

        with open('Result_Parameters/total_loss.txt', 'rb') as f:
            loss = np.load(f, allow_pickle=True)

        return loss

    def Plot(self):
        loss = self.ReadParameters()

        plt.plot(loss, label="training loss")
        plt.xlabel('epoch (x100)')
        plt.ylabel('smooth loss')
        plt.legend()
        plt.savefig('graph.png')
        plt.show()

if __name__ == '__main__':

    datafile = './Datasets/goblet_book.txt'
    n_epochs = 4
    retrain = True

    dataset = DatasetManager(datafile)

    rnn = RNN(dataset)
    if retrain:
        rnn.Train(n_epochs)

    rnn.Plot()
