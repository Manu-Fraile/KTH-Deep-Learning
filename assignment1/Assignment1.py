from functions import *

if __name__ == "__main__":

    training = LoadBatch('cifar-10-batches-py/data_batch_1')
    trainingData = training[b'data']
    trainingLabels = training[b'labels']

    validation = LoadBatch('cifar-10-batches-py/data_batch_2')
    validationData = validation[b'data']
    validationLabels = validation[b'labels']

    testing = LoadBatch('cifar-10-batches-py/test_batch')

    # print(training)
    print(training.keys())
    print('\n')
    print(training[b'data'])
    print(len(training[b'data']))
    print(len(training[b'data'][0]))
    print('\n')
    print(training[b'labels'])
    print(len(training[b'labels']))
