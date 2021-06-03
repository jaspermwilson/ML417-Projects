#!/usr/bin/python3
# Jasper Wilson
# jaspermwilson@wustl.edu
# CSE 417
# Homework 3 Code, using L1 and L2 regularizers for gradient descent
import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
import statistics
import matplotlib.pyplot as plt

def find_binary_error(w, X, y):
    # Inputs:
    # w: weight vector
    # X: data
    # y: labels
    # iterate through each datapoint and if the sign of the product of the
    #   weight vector and the datapoint is not the same as the label add 1 to
    #   the binary error before normalizing
    N, num_cols = X.shape
    total = 0
    for i in range(N):
        if(np.sign(np.matmul(w.T,X[i]))!=np.sign(y[i])):
            total = total + 1
    binary_error = total/N

    return binary_error


def find_ce_error(w, X, y):
    # Inputs:
    # w: weight vector
    # X: data
    # y: labels
    # compute the cross-entropy error of a linear classifier w
    #     on data set (X, y), this uses vectorization from numpy for efficiency
    N, num_cols = X.shape
    data_res = np.zeros([N,1])
    for i in range(N):
        data_res[i] = np.log(1 + np.exp(-y[i]*np.matmul(w.T,X[i])))

    final = np.mean(data_res)


    return final

def find_ce_grad(w, X, y):
    # Inputs:
    # w: weight vector
    # X: data
    # y: labels
    # compute the gradient of cross-entropy error of a linear classifier w on
    #    data set (X, y), again uses vectorization
    N, num_cols = X.shape

    # equation for ce error broken down for testing
    z = (y*X)
    ywX = y*(np.matmul(w.T,X.T).T)
    e = 1+np.exp(ywX)
    fin = z/e
    fin2 = 1 / N * np.sum(fin, axis = 0)

    #reshape so iteration works
    return fin2.reshape(65, 1)

def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # Inputs:
    # X: data
    # y: labels
    # w_init: initialized weight vector
    # max_its: maximum iterations before program quits
    # eta: learning rate
    # grad_threshold: if gradient drops below this point program will quit

    #initialize values for storing results
    t = 0
    w = w_init
    N, num_cols = X.shape
    grad = np.ones(num_cols)

    #iterate until we reach maximum iterations
    while (t < max_its):

        #calculate gradient, if it is less than the threshold break
        grad = find_ce_grad(w,X,y)
        if np.all(abs(grad) < grad_threshold):
            break
        #update weights
        w = w + eta*grad
        t = t+1

    e_in = find_ce_error(w, X, y)
    return t, w, e_in

def logistic_reg_L2(X, y, w_init, max_its, eta, grad_threshold, lambda_init, X_test, y_test):
    # Calculate final weights and errors using the L2 regularizer for gradient
    #    descent
    # Inputs:
    # X: train data
    # y: train labels
    # w_init: initialized weight vector
    # max_its: maximum iterations before program quits
    # eta: learning rate
    # grad_threshold: if gradient drops below this point program will quit
    # lambda_init: regulariztion strength
    # X_test: test data
    # y_test: test labels

    #instantiate variables
    t = 0
    w = w_init
    N, num_cols = X.shape
    grad = np.ones(num_cols)

    #iterate until max iterations is reached
    while (t < max_its):

        #calculate gradient
        grad = find_ce_grad(w,X,y)
        #break if all gradients are less than the threshold
        if np.all(abs(grad) < grad_threshold):
            break
        #update weights
        w = w*(1-2*lambda_init*eta) - eta*grad
        t = t+1

    #calculate binary errors for test and train data, count zeros
    e_in = find_binary_error(w, X, y)
    be_out = find_binary_error(w, X_test, y_test)
    num_zeros = w.size - np.count_nonzero(w)
    return lambda_init, be_out, num_zeros

def logistic_reg_L1(X, y, w_init, max_its, eta, grad_threshold, lambda_init, X_test, y_test):
    # Calculate final weights, errors, and number of zeros using the L2
    #     regularizer for gradient descent
    # Inputs:
    # X: train data
    # y: train labels
    # w_init: initialized weight vector
    # max_its: maximum iterations before program quits
    # eta: learning rate
    # grad_threshold: if gradient drops below this point program will quit
    # lambda_init: regulariztion strength
    # X_test: test data
    # y_test: test labels

    #instantiate variables
    t = 0
    w = w_init
    N, num_cols = X.shape
    grad = np.ones(num_cols)

    #iterate until max iterations is reached
    while (t < max_its):

        #break if all gradients are smaller than the threshold
        grad = find_ce_grad(w,X,y)
        if np.all(abs(grad) < grad_threshold):
            break
        #update weights, truncate to 0 if it makes the new weight change signs
        w_p = w - eta*grad
        w = w_p - lambda_init * eta * (np.sign(w))
        w[(np.sign(w_p) != np.sign(w))] = 0
        t = t+1

    #calculate errors and number of zeros
    e_in = find_binary_error(w, X, y)
    be_out = find_binary_error(w, X_test, y_test)
    num_zeros = np.count_nonzero(w == 0)
    return lambda_init, be_out, num_zeros


def main():
    #load data
    X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)
    #change zeros to -1s for this implementation of gradient descent to work
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    #normilize the testing training data
    train_x_norm = np.divide((X_train - np.mean(X_train, axis=0)),  np.std(X_train, axis=0), out=np.zeros_like((X_train - np.mean(X_train, axis=0))), where=np.std(X_train, axis=0)!=0)
    test_x_norm = np.divide((X_test - np.mean(X_train, axis=0)), np.std(X_train, axis=0), out=np.zeros_like((X_test - np.mean(X_train, axis=0))), where=np.std(X_train, axis=0)!=0)

    #reshape the labels so they work with iterations
    train_y_np = np.reshape(y_train, [len(y_train),1])
    test_y_np = np.reshape(y_test, [len(y_test),1])

    #append column of 1 to the front of x test to represent b as 1*w1
    num_rows, num_cols = test_x_norm.shape
    onesx = np.ones([num_rows,1])
    test_x_norm = np.append(onesx, test_x_norm, 1)

    #append column of 1 to the front of x train to represent b as 1*w1
    num_rows, num_cols = train_x_norm.shape
    onesx = np.ones([num_rows,1])
    train_x_norm = np.append(onesx, train_x_norm, 1)
    w = np.zeros([num_cols+1,1])

    #print the results of testing different levels of regularization strength
    print("L1s")
    print(logistic_reg_L1(train_x_norm,train_y_np,w,10000,0.01,0.000001, 0, test_x_norm, test_y_np))
    w = np.zeros([num_cols+1,1])
    print(logistic_reg_L1(train_x_norm,train_y_np,w,10000,0.01,0.000001, .0001, test_x_norm, test_y_np))
    w = np.zeros([num_cols+1,1])
    print(logistic_reg_L1(train_x_norm,train_y_np,w,10000,0.01,0.000001, .001, test_x_norm, test_y_np))
    w = np.zeros([num_cols+1,1])
    print(logistic_reg_L1(train_x_norm,train_y_np,w,10000,0.01,0.000001, .005, test_x_norm, test_y_np))
    w = np.zeros([num_cols+1,1])
    print(logistic_reg_L1(train_x_norm,train_y_np,w,10000,0.01,0.000001, .01, test_x_norm, test_y_np))
    w = np.zeros([num_cols+1,1])
    print(logistic_reg_L1(train_x_norm,train_y_np,w,10000,0.01,0.000001, .05, test_x_norm, test_y_np))
    w = np.zeros([num_cols+1,1])
    print(logistic_reg_L1(train_x_norm,train_y_np,w,10000,0.01,0.000001, .1, test_x_norm, test_y_np))
    w = np.zeros([num_cols+1,1])
    print("L2s")
    print(logistic_reg_L2(train_x_norm,train_y_np,w,10000,0.01,0.000001, 0, test_x_norm, test_y_np))
    print(logistic_reg_L2(train_x_norm,train_y_np,w,10000,0.01,0.000001, .0001, test_x_norm, test_y_np))
    print(logistic_reg_L2(train_x_norm,train_y_np,w,10000,0.01,0.000001, .001, test_x_norm, test_y_np))
    print(logistic_reg_L2(train_x_norm,train_y_np,w,10000,0.01,0.000001, .005, test_x_norm, test_y_np))
    print(logistic_reg_L2(train_x_norm,train_y_np,w,10000,0.01,0.000001, .01, test_x_norm, test_y_np))
    print(logistic_reg_L2(train_x_norm,train_y_np,w,10000,0.01,0.000001, .05, test_x_norm, test_y_np))
    print(logistic_reg_L2(train_x_norm,train_y_np,w,10000,0.01,0.000001, .1, test_x_norm, test_y_np))



if __name__ == "__main__":
    main()
