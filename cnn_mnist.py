from __future__ import absolute_import
# from __future__ import print_function
import numpy as np
import loader
import pandas as pd
import time as tm
np.random.seed(14)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

'''
    Train a simple convnet on the MNIST dataset.
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py
    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 12

img_rows, img_cols = 28, 28             # input image dimensions
nb_filters = 32                         # number of convolutional filters to use
nb_pool = 2                             # size of pooling area for max pooling
nb_conv = 3   #3                        # convolution kernel size

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = loader.load_data("mnist.pkl")
checkpoint = tm.time()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_real_test = pd.read_csv('test.csv').values
X_real_test = X_real_test.reshape(X_real_test.shape[0], 1, img_rows, img_cols)
X_real_test = X_real_test.astype("float32")
X_real_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
            init='he_normal', border_mode='valid',
            input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

print("Generating test predictions...")
preds = model.predict_classes(X_real_test, verbose=1)
write_preds(preds, "cnn_predictions.csv")

seconds = (tm.time() - checkpoint)
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print('----------- Finished fitting in %d:%02d:%02d -------------' % (h, m, s))