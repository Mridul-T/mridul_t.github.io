#%% --       PREREQS AND DEFINING FUNCTIONS THAT ARE GONNA BE USED
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as split
# Plotting library
from matplotlib import pyplot
from random import random
# Optimization module in scipy
from scipy import optimize
import seaborn as sn
from numba import jit, cuda
from sklearn.preprocessing import StandardScaler


# @jit(target_backend='cuda')
def softmax(z):
    ez=np.exp(z)
    a=ez/(np.sum(ez))
    return a
# @jit(target_backend='cuda')
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# @jit(target_backend='cuda')
def sigmoidGradient(z):
    """
    Computes the gradient of the sigmoid function evaluated at z. 
    
    Parameters
    ----------
    z : array_like : A vector or matrix as input to the sigmoid function. 
    
    Returns
    --------
    g : array_like : Gradient of the sigmoid function. Has the same shape as z. 
    
    """

    # ============================================================
    m = np.exp(-1*z)
    m = 1.0/(1+m)
    g=m*(1-m)
    # =============================================================
    return g
#%% --        PREPARING THE DATASET

X = pd.read_csv('X.csv')
X = StandardScaler().fit_transform(X)
y = pd.read_csv('y.csv')
# data.dropna(inplace=True)
# X=X.to_numpy(dtype=float)
y=pd.get_dummies(y,dtype=float)
print(y.head())
names=y.columns
y=y.to_numpy()
print(X.shape)
X_train, X_test, y_train, y_test = split(X, y, train_size=4000, shuffle=True)

#%% --           Setting up AND INTIALIZING the parameters

input_layer_size  = 2395 # no. of features in prepared dataset
hidden_layer1_size = 100 # 100 hidden units
hidden_layer2_size = 50 # 50 hidden units
num_labels = 19          # 19 labels, from 0 to 18

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer in a neural network.
    
    Parameters
    ----------
    L_in : int : Number of incomming connections.
    
    L_out : int : Number of outgoing connections. 
    
    epsilon_init : float, optional : Range of values which the weight can take from a uniform distribution.
    
    """
    W=np.empty(shape=(L_out,L_in))
    for i in range(L_out):
        for q in range(L_in):
            num = random() *epsilon_init * 2 - epsilon_init
            W[i][q]=num
    return W

w1 = randInitializeWeights(2396,100)
w2 = randInitializeWeights(101,50)
w3 = randInitializeWeights(51,19)

# Unroll parameters into a single array
initial_nn_params = np.concatenate([w1.ravel(),w2.ravel(),w3.ravel()], axis=0)

#%% --         IMPLEMENTING COST FUNCTION AND GRADIENT

# @jit(target_backend='cuda')
def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer1_size,
                   hidden_layer2_size,
                   num_labels,
                   X, y, lambda_=0.1):
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification. 
    
    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector.
    
    input_layer_size : int
        Number of features for the input layer. 
    
    hidden_layer1_size : int
        Number of hidden units in the second layer.
        
    hidden_layer2_size : int
        Number of hidden units in the third layer.
    
    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 
    
    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).
    
    y : array_like
        Dataset labels. A vector of shape (m,).
    
    lambda_ : float, optional
        Regularization parameter.
 
    Returns
    -------
    J : float : The computed value for the cost function at the current weight values.
    
    grad : array_like : An "unrolled" vector of the partial derivatives of the concatenatation of
                        neural network weights w1 and w2.
    """
    
    # Reshape nn_params back into the parameters w1 and w2, the weight matrices
    # for our 2 layer neural network
    w1 = np.reshape(nn_params[:hidden_layer1_size * (input_layer_size + 1)],
                        (hidden_layer1_size, (input_layer_size + 1)))

    w2 = np.reshape(nn_params[hidden_layer1_size * (input_layer_size + 1):hidden_layer1_size * (input_layer_size + 1) +
                              hidden_layer2_size * (hidden_layer1_size + 1)], (hidden_layer2_size, (hidden_layer1_size + 1)))
    
    w3 = np.reshape(nn_params[hidden_layer1_size * (input_layer_size + 1)+ hidden_layer2_size* (hidden_layer1_size + 1):],
                        (num_labels, (hidden_layer2_size + 1)))

    # Setup some useful variables
    m = y.shape[0]
    # print(m)
    X=np.concatenate([np.ones(shape=(m,1)),X],axis=1)
    # print(X.shape)
    # You need to return the following variables correctly 
    J = 0
    w1_grad = np.zeros(w1.shape)
    w2_grad = np.zeros(w2.shape)
    w3_grad = np.zeros(w3.shape)
    
    delta1_grad = np.zeros(w1.shape)
    delta2_grad = np.zeros(w2.shape)
    delta3_grad = np.zeros(w3.shape)

    # ================================================================================================
    
    regj=((np.sum(np.square(w1[:,1:]))+np.sum(np.square(w2[:,1:]))+np.sum(np.square(w3[:,1:])))/(2*m))*lambda_
    
    #calculating cost funtion
    
    for i in range(m):
        a1 = X[i,:][np.newaxis]
        a2 = sigmoid(np.dot(w1,np.transpose(a1)))
        a2 = np.concatenate((np.ones((1,1)),a2))
        a3 = sigmoid(np.dot(w2,a2))
        a3 = np.concatenate((np.ones((1,1)),a3))
        a4 = np.dot(w3,a3)
        hw=softmax(a4)
        Y = (y[i][np.newaxis]).T
        J+=np.sum(Y*np.log(hw))+np.sum((1-Y)*np.log(1-hw))
    
        delta4 = hw - Y
        delta = np.dot((np.transpose(w3)),delta4)
        delta3 = np.multiply(delta[1:],sigmoidGradient(np.dot(w2,a2)))
        delta = np.dot((np.transpose(w2)),delta3)
        delta2 = np.multiply(delta[1:],sigmoidGradient(np.dot(w1,np.transpose(a1))))               
        
        delta1_grad = delta1_grad + np.dot(delta2,a1)
        delta2_grad = delta2_grad + np.dot(delta3,np.transpose(a2))
        delta3_grad = delta3_grad + np.dot(delta4,np.transpose(a3))
        
        
    J=J/m*(-1)
    # Add regularization term
    J=J+regj

    # Backpropogation
    
    
    w1_grad = (1/m) * delta1_grad
    w2_grad = (1/m) * delta2_grad
    w3_grad = (1/m) * delta3_grad
    # Regularization of gradients
    # print(w2.shape)
    # print(w2_grad.shape)
    w1_grad[:, 1:] = w1_grad[:, 1:] + (lambda_ / m) * w1[:, 1:]
    w2_grad[:, 1:] = w2_grad[:, 1:] + (lambda_ / m) * w2[:, 1:]
    w3_grad[:, 1:] = w3_grad[:, 1:] + (lambda_ / m) * w3[:, 1:]

    #=======================================================================================================
    grad  = np.concatenate([w1_grad.ravel(), w2_grad.ravel(), w3_grad.ravel()],axis = 0)
    # print(J.shape)
    # print(grad.shape)
    return J, grad

nnCostFunction(initial_nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,X_train, y_train)

#%% --                                    OPTIMIZING

options= {'maxiter': 800}

#  You should also try different values of lambda
lambda_ = 0.25

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size, # p == nn_param
                                        hidden_layer1_size,hidden_layer2_size,
                                        num_labels, X_train, y_train, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
# epochs=10
# j=0
# while(j<epochs):
res=optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)
    # j+=1
# get the solution of the optimization
nn_params = res.x #Row vector containing weight values for every layer of your implemented neural network
        
# Obtain w1, w2 and w3 back from nn_params
w1 = np.reshape(nn_params[:hidden_layer1_size * (input_layer_size + 1)],
                        (hidden_layer1_size, (input_layer_size + 1)))

w2 = np.reshape(nn_params[hidden_layer1_size * (input_layer_size + 1):hidden_layer1_size * (input_layer_size + 1) +
                              hidden_layer2_size * (hidden_layer1_size + 1)], (hidden_layer2_size, (hidden_layer1_size + 1)))
w3 = np.reshape(nn_params[hidden_layer1_size * (input_layer_size + 1)+ hidden_layer2_size* (hidden_layer1_size + 1):],
                        (num_labels, (hidden_layer2_size + 1)))
print(w1)
print(w2)
print(w3)

#%% --             PREDICTING 

print(res)
def predict(w1, w2, w3, X):
    if X.ndim==1:
        X=X[None]
    m = X.shape[0]
    p = np.zeros(shape=(m,))
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    a2 = sigmoid(np.dot(w1,a1.T)).T
    a2 = np.concatenate([np.ones((m, 1)), a2], axis=1)
    a3 = sigmoid(np.dot(w2,a2.T)).T
    a3 = np.concatenate([np.ones((m, 1)), a3], axis=1)
    z4 = np.dot(w3,a3.T).T
    ans = np.zeros(shape=z4.shape)
    print(z4.shape)
    for i in range(z4.shape[0]):
        ans[i]=softmax(z4[i])
    # print(np.sum(ans))
    p = np.argmax(ans, axis=1)
    return p

pred_train = predict(w1, w2, w3, X_train)
pred_test = predict(w1, w2, w3, X_test)
# print(type(pred[0]))
# print(pred[:100])
Y_train=np.argmax(y_train, axis=1)
Y_test=np.argmax(y_test, axis=1)
# print(type(Y[0]))

# print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))
# %% ---     GETTING THE CONFUSION MATRIX

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
print('Training Set Accuracy: %f' % (float(accuracy_score(Y_train,pred_train))*100))
print('Test Set Accuracy: %f' % (float(accuracy_score(Y_test,pred_test))*100))

classes=np.arange(start=0,stop=19,step=1)
mat=confusion_matrix(Y_test, pred_test, labels=classes, sample_weight=None, normalize='true')
matdf=pd.DataFrame(mat, columns=names, index=names)
sn.heatmap(matdf)
