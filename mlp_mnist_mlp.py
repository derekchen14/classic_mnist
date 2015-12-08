import time as tm
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.visualize_util import plot
# from keras.optimizers import SGD

# Read data
checkpoint = tm.time()
train = pd.read_csv('train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')
y_train = np_utils.to_categorical(labels)
print "Loaded in %0.2f seconds" % (tm.time() - checkpoint)

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]  # 784
num_classes = y_train.shape[1]  # 10
init_type = 'glorot_normal'

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(input_dim=input_dim, output_dim=128, init=init_type))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(input_dim=128, output_dim=128, init=init_type))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(input_dim=128, output_dim=num_classes, init=init_type))
model.add(Activation('tanh'))

# we'll use MSE (mean squared error) for the loss, and RMSprop as the optimizer
# sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='rmsprop')

print("Training...")
model.fit(X_train, y_train, nb_epoch=42, batch_size=16, validation_split=0.1, show_accuracy=True, verbose=1)

plot(model, to_file='model.png')


print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=1)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "mlp_predictions.csv")