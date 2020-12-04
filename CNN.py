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
    test_images = params['test_images']
    test_filenames = params['filenames']
    
    num_train_images = params['num_train_images']
    num_test_images = params['num_test_images']
    image_width = params['image_width']
    stride = params['stride']
    
    #num_train_images = int(num_images * 4/5)
    #num_test_images = int(num_images * 1/5)
    
    try:
        for i in range(0, num_conv_layers):
            if i==0:
                common_features.append(Conv2D(num_filters, kernel_size=kernel_size, activation=conv_activation, input_shape=(image_width,image_width,1), strides=(stride, stride)))
            else:
                common_features.append(Conv2D(num_filters, kernel_size=kernel_size, activation=conv_activation, strides=(stride, stride)))
        
            if i<num_pool_layers:
                common_features.append(MaxPooling2D(pool_size = (pool_size, pool_size), strides=(stride, stride)))
               
        for i in range(0, num_dense_layers):
            if i==0:
                common_features.append(Flatten())
                 
            common_features.append(Dense(dense_layer_size, activation=dense_activation))
        common_features.append(Dense(2, activation='sigmoid'))
        
        
        
           
                
        cnn_model = Sequential(common_features)
        print(cnn_model.summary())
        cnn_model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'],)
        
        cnn_model.fit(np.reshape(train_images, [num_train_images, image_width, image_width, 1]), np.reshape(to_categorical(train_labels), [num_train_images, 2]), epochs=20, batch_size=16,)
        
        predictions = cnn_model.predict_classes(test_images, batch_size=1, verbose=0)
        majority_voted_predictions = []
        
        for i in range(0, len(predictions), 3):
            votes = predictions[i] + predictions[i+1] + predictions[i+2]
            if votes > 1:
                majority_voted_predictions.append(1)
            else:
                majority_voted_predictions.append(0)
        
        with open('cnn2_predictions.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, len(filenames), 3):
                csvwriter.writerow([filenames[i], predictions[int(i/3)]])
        
        
        return 
    
    except Exception as e:
        print("ERROR: {}".format(e))
        return 1000
        
        
        
train_images = []
train_labels = []
test_images = []
filenames = []

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
        g, gx, gy = fe.edge_detection(read_resized_image(filename, new_width))
        filtered_image_width = np.shape(g)[0]
        train_images.append(np.reshape(g, [filtered_image_width, filtered_image_width, 1]))
        train_images.append(np.reshape(gx, [filtered_image_width, filtered_image_width, 1]))
        train_images.append(np.reshape(gy, [filtered_image_width, filtered_image_width, 1]))
        train_labels.append(int(train_label))
        train_labels.append(int(train_label))
        train_labels.append(int(train_label))
        


first_line = 0
count2 = 0
with open('test.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if first_line < 1:
            first_line += 1
            continue
        
        data = row[0].split(',')
        test_label = row[len(row)-1].split(',')[len(row[len(row)-1].split(','))-1]
            
        filename = "test_images/"+data[0]
        filenames.append(data[0])
        filenames.append(data[0])
        filenames.append(data[0])
        count2+= 1
        g, gx, gy = fe.edge_detection(read_resized_image(filename, new_width))
        filtered_image_width = np.shape(g)[0]
        test_images.append(np.reshape(g, [filtered_image_width, filtered_image_width, 1]))
        test_images.append(np.reshape(gx, [filtered_image_width, filtered_image_width, 1]))
        test_images.append(np.reshape(gy, [filtered_image_width, filtered_image_width, 1]))

num_train_images = count1*3
num_test_images = count2*3

train_images = np.reshape(train_images, [num_train_images, filtered_image_width, filtered_image_width, 1])
test_images = np.reshape(test_images, [num_test_images, filtered_image_width, filtered_image_width, 1])

tpe_algo = tpe.suggest

#best CNN model
space = {
    'activ': 'softmax',
    'kern':  3,
    'nfilt': 20,
    'pool_size': 3,
    'dense_size': 256,
    'num_dense_layers': 1,
    'dense_activ': 'relu',
    'num_conv': 1,
    'num_pool': 0,
    'stride':  2,
    
    'train_images': train_images,
    'train_labels': train_labels,
    'test_images': test_images,
    
    'filenames': filenames,
    'fd': f,
    'num_train_images': num_train_images,
    'num_test_images': num_test_images,
    'image_width': filtered_image_width
}

#second best CNN model
space = {
    'activ': 'softmax',
    'kern':  4,
    'nfilt': 40,
    'pool_size': 3,
    'dense_size': 512,
    'num_dense_layers': 1,
    'dense_activ': 'relu',
    'num_conv': 1,
    'num_pool': 0,
    'stride':  3,
    
    'train_images': train_images,
    'train_labels': train_labels,
    'test_images': test_images,
    
    'filenames': filenames,
    'fd': f,
    'num_train_images': num_train_images,
    'num_test_images': num_test_images,
    'image_width': filtered_image_width
}

make_cnn(space)



