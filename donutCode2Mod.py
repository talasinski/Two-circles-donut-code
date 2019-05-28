# This basis of this code was presented in the Udemy course (https://www.udemy.com)
# "Data Science: Deep Learning in Python", by the Lazy Programmer
# The graphics with pseudo-animation was added by T.A.Lasinski (2019)
# The code has a one layer neural network with backpropagtion.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion, ioff, isinteractive, draw
import matplotlib.animation as animation
#from celluloid import Camera
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import random as R
# for binary classification! no softmax here

def forward(X, W1, b1, W2, b2):
    # sigmoid
    # Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))

    # tanh
    # Z = np.tanh(X.dot(W1) + b1)

    # relu
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)

    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z


def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)


def derivative_w2(Z, T, Y):
    # Z is (N, M)
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    return (T - Y).sum()


def derivative_w1(X, Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return X.T.dot(dZ)


def derivative_b1(Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return dZ.sum(axis=0)


def get_log_likelihood(T, Y):
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    R.seed(101)
    f= R.uniform(1.0-noise/2,1+noise/2)
    n_points =  n_points//2
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360 *f
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * f
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * f
    #n_points2 = int(n_points//2)
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))


def makeInput():
    N=2000
#    X, y = make_moons(N, noise=0.11)
#    X,y = twospirals(N, 0.2)
    X, y = make_circles(n_samples = N, random_state=123, noise=0.1, factor=0.2)
    plt.title("Complete Data Set")
    plt.scatter(X[:, 0], X[:, 1],c=y)
    plt.show()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.75, random_state=73)
    return X_train, X_test, Y_train, Y_test

def boundary(X, W1, b1, W2, b2): # determine boundary between different colored dots
	x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
	y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
	spacing = min(x_max - x_min, y_max - y_min) / 100
	XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),np.arange(y_min, y_max, spacing))
	data = np.hstack((XX.ravel().reshape(-1,1),YY.ravel().reshape(-1,1)))
	db_prob =predict(data, W1, b1, W2, b2)
	clf = np.where(db_prob<0.5,0,1)
	Z = clf.reshape(XX.shape)
	return(plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.6))


def main():

    X_train, X_test, Y_train, Y_test = makeInput()
    X = X_train
    Y = Y_train
    print(X_test.shape, Y_test.shape)
    print(len(X_test), len(Y_test))
    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.title("Train Data Set")
    plt.show()
    n_hidden = 50
    iter = 100
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = [] # keep track of log-likelihoods
    learning_rate = 0.001
    regularization = 0.2
    last_error_rate = None

#    camera = Camera(plt.figure())
    for i in range(iter):

        pY, Z = forward(X, W1, b1, W2, b2)
        ll = get_log_likelihood(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        predictionTest = predict(X_test, W1, b1, W2, b2)
        erTrain = np.abs(prediction - Y).mean()
        erTest = np.abs(predictionTest - Y_test).mean()
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
#        	print(prediction - Y)
        if i % 1 == 0:
        	print("i:", i, "ll: {:2.3f}".format( ll), "classification rate: {:0.3f}".format(1 - erTrain))
        	plt.figure(figsize=(8,4))
        	plt.subplot(1,2,1)
        	_ = boundary(X,W1, b1, W2, b2)
        	string = "Train Acc: {:0.3f}".format((1 - erTrain))
        	plt.title(string)

        	plt.scatter(X[:,0], X[:,1], c=prediction)
        	plt.subplot(1,2,2)
        	_ = boundary(X_test,W1, b1, W2, b2)
        	string = "Test  Acc: {:0.3f}".format((1 - erTest))
        	plt.title(string)

        	plt.scatter(X_test[:,0], X_test[:,1], c=predictionTest)
        	plt.tight_layout()

        	plt.show(block=False)
        	plt.pause(0.4)
        	plt.close()
#        	camera.snap()

    f = plt.figure(figsize=(8,6))
    ax = f.add_subplot(1,2,1)
    string = "Train Acc: {:0.3f}".format((1 - erTrain))
    plt.title(string)
    ax.scatter(X[:,0], X[:,1], c=prediction)
    ax2=f.add_subplot(1,2,2)
    string = "Test Acc: {:0.3f}".format((1 - erTest))
    plt.title(string)

    ax2.scatter(X_test[:,0], X_test[:,1], c=predictionTest)

    plt.show()
    plt.plot(LL)
    plt.show()

    print('Iteratioins: ', iter, 'Neurons: ', n_hidden,' Learning rare: ',learning_rate, ' Regularization: ',regularization)

if __name__ == '__main__':
    main()

