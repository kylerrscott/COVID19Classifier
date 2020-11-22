from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal
from skimage import feature
from sklearn.decomposition import PCA
import csv
import matplotlib.pyplot as plt
import statistics
import scipy.stats
from keras.utils import to_categorical
import random
import sys
import time
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from hyperopt import tpe, hp, fmin, Trials
start = time.time()



# called by hyperopt, constructs and tests a CNN based on the values in params
# measures and returns avg 5-fold cross-validation classification accuracy
def make_cnn(params):
    
    common_features = []
    conv_activation = params['activ']
    kernel_size = params['kern']
    num_filters = params['nfilt']
    pool_size = params['pool_size']
    dense_layer_size = params['dense_size']
    num_dense_layers = params['num_dense_layers']
    dense_activation = params['dense_activ']
    num_conv_layers = params['num_conv']
    num_pool_layers = params['num_pool']
    
    train_images = params['train_images']
    train_labels = params['train_labels']
    num_images = params['num_images']
    image_width = params['image_width']
    stride = params['stride']
    
    num_train_images = int(num_images * 4/5)
    num_test_images = int(num_images * 1/5)
    
    try:
        #construct CNN
        
        for i in range(0, num_conv_layers):
            if i==0:
                common_features.append(Conv2D(num_filters, kernel_size=kernel_size, activation=conv_activation, input_shape=(image_width,image_width,1), strides=(stride, stride)))
            else:
                common_features.append(Conv2D(num_filters, kernel_size=kernel_size, activation=conv_activation, strides=(stride, stride)))
        
            if i<num_pool_layers:
                common_features.append(MaxPooling2D(pool_size = (pool_size, pool_size), strides=(stride, stride)))
            
            if kernel_size > 3:
                kernel_size -= 1
               
        for i in range(0, num_dense_layers):
            if i==0:
                common_features.append(Flatten())
                 
            common_features.append(Dense(dense_layer_size, activation=dense_activation))
        common_features.append(Dense(2, activation='sigmoid'))
        
        
        # measure 5-fold classification accuracy
        
        num_folds = 5
        kfold = StratifiedKFold(n_splits = num_folds, shuffle=True, random_state=1)
        kfold.get_n_splits(train_images, train_labels)
        average_train_performance = 0
        
        for train_indices, test_indices in kfold.split(train_images, train_labels):
            folded_train_images = []
            folded_train_labels = []
            folded_test_images = []
            folded_test_labels = []
            
            for train_index in train_indices:
                folded_train_images.append(train_images[train_index])
                folded_train_labels.append(train_labels[train_index])
                
            for test_index in test_indices:
                folded_test_images.append(train_images[test_index])
                folded_test_labels.append(train_labels[test_index])
                
            cnn_model = Sequential(common_features)
            print(cnn_model.summary())
            cnn_model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'],)
            
            cnn_model.fit(np.reshape(folded_train_images, [num_train_images, image_width, image_width, 1]), np.reshape(to_categorical(folded_train_labels), [num_train_images, 2]), epochs=20, batch_size=16,)
            
            # measure classification accuracy for the validation fold
            train_performance = cnn_model.evaluate(np.reshape(folded_test_images, [num_test_images, image_width, image_width, 1]), np.reshape(to_categorical(folded_test_labels), [num_test_images, 2]))
            average_train_performance += train_performance[1]
            print("Accuracy on Train set: {0}".format(train_performance[1]))
        
        # return avg classification accuracy over 5 folds
        average_train_performance /= 5
        return average_train_performance*-1
    
    # if an error happens, return a big number. this can be triggered sometimes, for instance if there are more pool layers than conv layers
    except Exception as e:
        print("ERROR: {}".format(e))
        return 1000


# reading in training images and using hyperopt

train_images = []
train_labels = []

first_line = 0
new_width = 128
count1 = 0

fe = FeatureExtract()
filtered_image_width = 1
with open('train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if first_line < 1:
            first_line += 1
            continue
        
        
        data = row[0].split(',')
        train_label = row[len(row)-1].split(',')[len(row[len(row)-1].split(','))-1]
            
        filename = "train_images/"+data[0]
        count1+= 1
        
        # reads in a training image, adds the output of sobel filter to the set used for CNN training
        
        resized_image = read_resized_image(filename, new_width)
        g, gx, gy = fe.edge_detection(resized_image)
        filtered_image_width = np.shape(g)[0]
        train_images.append(np.reshape(g, [filtered_image_width, filtered_image_width, 1]))
        train_images.append(np.reshape(gx, [filtered_image_width, filtered_image_width, 1]))
        train_images.append(np.reshape(gy, [filtered_image_width, filtered_image_width, 1]))
        train_labels.append(int(train_label))
        train_labels.append(int(train_label))
        train_labels.append(int(train_label))


# tells us how many training images we have
num_images = count1*3


tpe_algo = tpe.suggest

space = {
    'activ': hp.choice('a', ['softmax', 'relu', 'sigmoid']),
    'kern': hp.choice('k', [3, 4, 5, 7, 9]),
    'nfilt': hp.choice('nf', [20, 30, 40, 50, 60, 70]),
    'pool_size': hp.choice('ps', [2, 3]),
    'dense_size': hp.choice('cs', [128, 256, 512, 1024]),
    'num_dense_layers': hp.choice('ndl', [1, 2, 3, 4]),
    'dense_activ': hp.choice('da', ['softmax', 'relu', 'sigmoid']),
    'num_conv': hp.choice('nc', [1, 2, 3, 4]),
    'num_pool': hp.choice('np', [0, 1, 2]),
    'stride': hp.choice('str', [1, 2, 3]),
    
    'train_images': train_images,
    'train_labels': train_labels,
    'fd': f,
    'num_images': num_images,
    'image_width': filtered_image_width
}

tpe_trials = Trials()
tpe_best = fmin(fn=make_cnn, space=space, algo=tpe_algo, trials=tpe_trials, max_evals = 50)
print("tpe best {}".format(tpe_best))







