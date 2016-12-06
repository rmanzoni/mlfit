import numpy as np

from lasagne import layers

from nolearn.lasagne import NeuralNet    
from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split

# create a training dataset
X, y, coef = make_regression(n_samples=1000, n_features=30, noise=0.15, coef=True, random_state=1986)
y = y.reshape(-1, 1).astype(np.float32)

# split test/train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1986)

print 'input shape (samples, features)', X_train.shape
print 'target shape (samples,)'        , y_train.shape

l = layers.InputLayer(shape=(None, X_train.shape[1]))
l = layers.DenseLayer(l, num_units=y_train.shape[1], nonlinearity=None)
l = layers.DenseLayer(l, num_units=y_train.shape[1], nonlinearity=None)

net = NeuralNet(
    layers=l, 
    regression=True, 
    update_learning_rate=0.001,
    verbose=99,
    max_epochs=100,
)

net.fit(X_train, y_train)

y_pred = net.predict(X_test)
print "The accuracy of this network is: %0.5f percent" % (abs(1. - y_pred/y_test)*100.).mean()
