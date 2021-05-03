from Assignment3 import *


def read_file(filename, validation_filename):
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
