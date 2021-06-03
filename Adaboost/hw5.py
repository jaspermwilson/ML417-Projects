#!/usr/bin/python3
# Jasper Wilson
# jaspermwilson@wustl.edu
# CSE 417
# Homework 5 Code, Adaboost on digit distintions using pixel dataset
# Runtime: 12 minutes
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import math


def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # AdaBoost: Implement AdaBoost using decision trees
    #    using information gain as the weak learners.
    #    X_train: Training set
    #    y_train: Training set labels
    #    X_test: Testing set
    #    y_test: Testing set labels
    #    n_trees: The number of trees to use

    #initialize normalized array of weights for each item
    weights = np.empty(np.shape(X_train)[0])
    weights.fill(1)
    weights = weights/np.sum(weights)

    #initialize empty arrays to store errors from each trial and item
    #    where each row is an item and each column is a tree
    train_errs = np.empty([np.shape(X_train)[0],n_trees])
    test_errs = np.empty([np.shape(X_test)[0],n_trees])

    #set parameters for the classifier
    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 1)

    #iterate through the number of trees, storing the weighted error
    #    for each item
    for i in range (0, n_trees):
        #fit a tree with the weights for each item
        clf.fit(X_train, y_train, weights)

        #save predictions for each item in test and train data
        pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        #save binary error
        error_matrix = np.not_equal(pred, y_train)
        #reweigh the errors
        error = np.sum(error_matrix*weights)
        #calculate alpha, a measure of goodness of this hypothesis used to weigh
        #    this tree
        alpha = math.log((1-error)/error)/2
        # store the weighted predictions for each datapoint for the given tree
        train_errs[:,i] = alpha*pred
        test_errs[:,i] = alpha*test_pred
        # update and normalize weights
        weights = weights*np.exp((-1)*alpha*pred*y_train)
        weights = weights/np.sum(weights)


    #sum the weighted prediction for each tree for every datapoint, use sign
    #    function to get prediction for each point, then calculate mean binary
    #    error
    train_error = np.sum(np.not_equal(np.sign(np.sum(train_errs, axis = 1)), y_train))/np.shape(X_train)[0]
    test_error = np.sum(np.not_equal(np.sign(np.sum(test_errs, axis = 1)), y_test))/np.shape(X_test)[0]

    return train_error, test_error


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')



    # Split 1,3 data
    og_train_data_13 = og_train_data[np.where((og_train_data[:,0]==1)|(og_train_data[:,0]==3))]
    og_test_data_13 = og_test_data[np.where((og_test_data[:,0]==1)|(og_test_data[:,0]==3))]

    #Split train and test data, first column is y values
    X_train_13 = og_train_data_13[:,1:257]
    y_train_13 = og_train_data_13[:,0]
    X_test_13 = og_test_data_13[:,1:257]
    y_test_13 = og_test_data_13[:,0]

    #Convert 3s to -1s for the meaning 1 is positive case and 3 is the
    #    negative case, necessary transform for this
    #    version of the Adaboost algorithm
    y_train_13 = np.where(y_train_13 == 3, -1, y_train_13)
    y_test_13 = np.where(y_test_13 == 3, -1, y_test_13)

    #repeat the steps above for 3,5 data
    og_train_data_35 = og_train_data[np.where((og_train_data[:,0]==3)|(og_train_data[:,0]==5))]
    og_test_data_35 = og_test_data[np.where((og_test_data[:,0]==3)|(og_test_data[:,0]==5))]

    #Split train and test data, first column is y values
    X_train_35 = og_train_data_35[:,1:257]
    y_train_35 = og_train_data_35[:,0]
    X_test_35 = og_test_data_35[:,1:257]
    y_test_35 = og_test_data_35[:,0]

    #Convert 3s to -1s and 5s to +1, necesarry transform for this version of
    #    Adaboost
    y_train_35 = np.where(y_train_35 == 3, -1, y_train_35)
    y_train_35 = np.where(y_train_35 == 5, 1, y_train_35)

    y_test_35 = np.where(y_test_35 == 3, -1, y_test_35)
    y_test_35 = np.where(y_test_35 == 5, 1, y_test_35)

    #Create arrays to store test and train errors for 1-200 trees
    #    for plotting error as a function of number of trees
    train_array_35 = np.zeros(200)
    test_array_35 = np.zeros(200)
    train_array_13 = np.zeros(200)
    test_array_13 = np.zeros(200)
    #Loop from 1-200 trees
    for j in range (0,200):
        num_trees = j + 1
        train_array_13[j], test_array_13[j] = adaboost_trees(X_train_13, y_train_13, X_test_13, y_test_13, num_trees)
        train_array_35[j], test_array_35[j] = adaboost_trees(X_train_35, y_train_35, X_test_35, y_test_35, num_trees)

    #plots
    plt.figure()
    plt.plot(train_array_13)
    plt.plot(test_array_13)
    plt.legend(["Train Error","Test Error"])
    plt.title("1 vs 3 Errors")
    plt.xlabel("Number of Trees")
    plt.ylabel("Error")

    plt.savefig("13plot")
    plt.show

    plt.figure()
    plt.plot(train_array_35)
    plt.plot(test_array_35)
    plt.legend(["Train Error","Test Error"])
    plt.title("3 vs 5 Errors")
    plt.xlabel("Number of Trees")
    plt.ylabel("Error")

    plt.savefig("35plot")
    plt.show



if __name__ == "__main__":
    main_hw5()
