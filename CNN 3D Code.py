import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Dense, Flatten
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.regularizers import l2
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Preparing the data
X_train = sio.loadmat('./Data/dB/step_130/X_train_balanced_contineous')['X_train']
Y_train = sio.loadmat('./Data/dB/step_130/Y_train_balanced_contineous')['targets_train']
X_train = np.expand_dims(X_train, axis=4)

X_test = sio.loadmat('./Data/dB/step_130/X_test_balanced_contineous')['X_test']
Y_test = sio.loadmat('./Data/dB/step_130/Y_test_balanced_contineous')['targets_test']
X_test = np.expand_dims(X_train, axis=4)
Y_test = np.squeeze(Y_test)
Y_test = to_categorical(Y_test,num_classes=4).astype(np.integer)

# Shuffle the training data
# fix random seed for reproducibility
seedValue = 40
permutation = np.random.RandomState(seed=seedValue).permutation(len(X_train))
X_train = X_train[permutation]
Y_train = Y_train[permutation]
Y_train = np.squeeze(Y_train)
# Determine sample shape
sample_shape = (64, 64, 38, 1)

# Model configuration
batch_size = 32
no_epochs = 50
learning_rate = 0.001
no_classes = 4
verbosity = 1

pool_size = (2,2,2)
l2_lambda = 5*pow(10,-4)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5)
cvscores = []
condition = 1

hist=np.zeros((5,4,no_epochs))
count = 0

for train, test in kfold.split(X_train, Y_train):
    if condition == 1:
        Y_train = to_categorical(Y_train,num_classes=4).astype(np.integer)
        condition = 0
    # Create the model
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3,3,3),  kernel_regularizer=l2(l2_lambda), activation='selu', padding='same', kernel_initializer='glorot_uniform', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=pool_size))
    #model.add(Dropout(0.25))

    model.add(Conv3D(16, kernel_size=(3,3,3),  kernel_regularizer=l2(l2_lambda), activation='selu', padding='same', kernel_initializer='glorot_uniform', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=pool_size))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(l2_lambda), activation='selu', kernel_initializer='glorot_uniform'))
    #model.add(Dropout(0.5))

    model.add(Dense(no_classes, activation='softmax'))
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=learning_rate),
                metrics=['accuracy'])
    # Fit data to model
    history = model.fit(X_train[train], Y_train[train],
                batch_size=batch_size,
                epochs=no_epochs,
                validation_data=(X_train[test], Y_train[test]),
                verbose=verbosity)

    hist[count,0] = history.history['loss']
    hist[count,1] = history.history['val_loss']
    hist[count,2] = history.history['accuracy']
    hist[count,3] = history.history['val_accuracy']
    count = count+1

    # Generate generalization metrics
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
sio.savemat('history.mat', {'history':hist})

# Plot history: Categorical crossentropy & Accuracy
plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Model performance for 3D Keras Conv3D')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
