#!/usr/bin/python3
# Jasper Wilson
# jaspermwilson@wustl.edu
# CSE 417
# Homework 4 Code, random forest bagging on digit distintions using pixel dataset
# Runtime: 50 minutes, could be optimized by only running 200 bags once and then
#    calulating error as different sizes from those 200 bags
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # Inputs:
    # `X_train` is the training data
    # `y_train` are the training labels
    # `X_test` is the testing data
    # `y_test` are the testing labels
    # `num_bags` is the number of trees to learn in the ensemble
    #
    # Outputs:
    # `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # `test_error` is the classification error of the final learned ensemble on test data

    #Create empty lists to store
    test_preds = np.zeros((len(X_test),num_bags))
    OOB_array = np.empty((len(X_train),num_bags))
    OOB_array[:] = np.nan
    total_index = range(0,len(X_train))
    OOB_COUNT = 0;
    clf = DecisionTreeClassifier(criterion = "entropy")
    for i in range(num_bags):
        #generate a bag, which takes n random indexes from an array of length
        #    n sampling with replacement to create smaller random dataset
        rand_indexs = np.random.randint(0, len(X_train), len(X_train))
        #strap dataset
        strapped_X = X_train[rand_indexs, :]
        strapped_y = y_train[rand_indexs]
        #store unstrapped indexes for calculating out of bag error
        unstrapped_indexes = np.setdiff1d(total_index, rand_indexs)

        #fit decision tree with strapped data
        clf.fit(strapped_X, strapped_y)


        #Save predictions on test data for test error
        test_preds[:,i] = clf.predict(X_test);
        #Iterate through array of indexes and add them to OOB array with columns
        #    being each tree and rows being each data point. If an index
        #    is included in the strapped data it will be nan
        for j in unstrapped_indexes:
            OOB_array[j,i] = clf.predict(X_train[j].reshape(1,-1))



    #Calculate OOB error
    OOB_SUM = 0;
    for k in range(0,len(OOB_array)):
        # Use mode to make final prediction, exclude nans
        mode, count = stats.mode(OOB_array[k], nan_policy="omit")
        #include this to account for edge cases where a datapoint is included
        #    in every bag, in which case the mode will be 0
        if(mode != 0):
            OOB_SUM += (mode == y_train[k])
            OOB_COUNT += 1
    #Calculate the % error, accounting for any points included in all bags
    out_of_bag_error = 1 - OOB_SUM/OOB_COUNT

    #Calculate test error using mode, no need to worry about bags
    test_sum = 0;
    for l in range(0,len(test_preds)):
        mode2, count2 = stats.mode(test_preds[l])
        test_sum += (mode2 == y_test[l])
    #calculate the % test error
    test_error = 1 - test_sum/len(test_preds)

    return out_of_bag_error, test_error

# return error for single decision tree, used for testing
def single_decision_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion = "entropy")
    clf.fit(X_train, y_train)
    train_error = 1- clf.score(X_train, y_train)
    test_error = 1 - clf.score(X_test, y_test)
    return train_error,test_error

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    # seperate data where digits are 1 and 3
    og_train_data_13 = og_train_data[np.where((og_train_data[:,0]==1)|(og_train_data[:,0]==3))]
    og_test_data_13 = og_test_data[np.where((og_test_data[:,0]==1)|(og_test_data[:,0]==3))]

    # split data into test and train data, seperate y from parameters
    X_train_13 = og_train_data_13[:,1:257]
    y_train_13 = og_train_data_13[:,0]
    X_test_13 = og_test_data_13[:,1:257]
    y_test_13 = og_test_data_13[:,0]

    # seperate data where digits are 3 and 5
    og_train_data_35 = og_train_data[np.where((og_train_data[:,0]==3)|(og_train_data[:,0]==5))]
    og_test_data_35 = og_test_data[np.where((og_test_data[:,0]==3)|(og_test_data[:,0]==5))]

    #split data into test and train data, seperate y from parameters
    X_train_35 = og_train_data_35[:,1:257]
    y_train_35 = og_train_data_35[:,0]
    X_test_35 = og_test_data_35[:,1:257]
    y_test_35 = og_test_data_35[:,0]


    #Create empty arrays to store errors for number of bags 1-200
    errors_13 = np.zeros(200)
    errors_35 = np.zeros(200)
    #loop through number of bags 1-200, storing out of bag and test errors of
    #    final forests
    for i in range (1,201):
        out_of_bag_error_13, test_error_13 = bagged_trees(X_train_13, y_train_13, X_test_13, y_test_13, i)
        out_of_bag_error_35, test_error_35 = bagged_trees(X_train_35, y_train_35, X_test_35, y_test_35, i)
        errors_13[i-1] = out_of_bag_error_13
        errors_35[i-1] = out_of_bag_error_35

    #plots
    plt.figure()
    plt.plot(errors_13)
    plt.title("OOB error of 1 vs 3 distinction")
    plt.xlabel("Number of Bags")
    plt.ylabel("OOB error")
    plt.savefig("13plot")
    plt.show

    plt.figure()
    plt.plot(errors_35)
    plt.title("OOB error of 3 vs 5 distinction")
    plt.xlabel("Number of Bags")
    plt.ylabel("OOB error")
    plt.savefig("13plot")
    plt.show

    #print errors for num bags = 200
    print("1 vs 3 OOB error:")
    print(out_of_bag_error_13)
    print("1 vs 3 test error:")
    print(test_error_13)
    print("3 vs 5 OOB error:")
    print(out_of_bag_error_35)
    print("3 vs 5 test error:")
    print(test_error_35)






if __name__ == "__main__":
    main_hw4()
