import numpy as np

def read_data(file_name):
    return np.loadtxt(file_name, delimiter=',')

def one_hot_labels(labels):
    '''
        labels = [5., 0., 4., ..., 8.]

        return labels \in [labels.shape[0], 10]
    '''
    res = np.zeros((labels.shape[0], 10))
    for i, l in enumerate(labels):
        res[i][int(l)] = 1
    return res

def get_data(path='./data/dataset/'):
    print('start reading data...')

    train_data = read_data(path + 'images_train.csv')
    train_labels = one_hot_labels(read_data(path + 'labels_train.csv'))

    test_data = read_data(path + '/images_test.csv')
    test_labels = one_hot_labels(read_data(path + 'labels_test.csv'))

    print('finish reading data!')
    return train_data, train_labels, \
            test_data, test_labels
    

