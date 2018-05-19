"""
Scaffolding code for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes
as necessary.

Write a main function that calls the different functions to perform the
required tasks
and repeat your experiments.
"""

"""
Helpful resources

http://scikit-learn.org/stable/tutorial/basic/tutorial.html

loading datasets: http://scikit-learn.org/stable/datasets/index.html
"""

import csv
import numpy as np


def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    """
    return [(9821112, 'Jonathan', 'Gonzalez'), (8850224, 'Rachel', 'Lynch')]


def prepare_dataset(dataset_path):
    """
    Read a comma separated text file where
    - the first field is a ID number
    - the second field is a class label 'B' or 'M'
    - the remaining fields are real-valued

    Return two numpy arrays X and y where
    - X is two dimensional. X[i,:] is the ith example
    - y is one dimensional. y[i] is the class label of X[i,:]
        y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the data set text file

    @return: X,y
    """
    # https://docs.python.org/2.3/whatsnew/node14.html
    # - how to read data file in
#    input = open('/Users/JNGZ/PycharmProjects/cab302_ML/medical_records.data'
#                 ,'rt')
    input = open(dataset_path, 'rt')    # input param is dataset_path
    reader = csv.reader(input)
    tumor_index = 1  # index where rating 'M' or 'B' is located in the data row
    

    # Create empty python helper array to build X
    array = []

    # iterate reader and append each line from the data set to the empty
    # python array
    for line in reader:
        array.append(line)

    X = np.array(array)  
    
    # Empty the helper array to build y
    array = []
    
    # for each row in X, get the value of index 1 and assign it 0 for 'B' and '1' for M
    for i in range(X.shape[0]):
        tumor_rating = X[i][tumor_index]
        if tumor_rating == 'B':
            tumor_rating = 0
        elif tumor_rating == 'M':
            tumor_rating = 1
        else:
            tumor_rating = "Invalid Rating"
        array.append(tumor_rating)
        
    y = np.array(array)

    return X, y


def build_NB_classifier(X_training, y_training):
    """
    Build a Naive Bayes classifier based on the training set X_training,
    y_training.

    @param X_training, y_training:
    X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return clf : the classifier built in this function
    """

    raise NotImplementedError()


def build_DT_classifier(X_training, y_training):
    """
    Build a Decision Tree classifier based on the training set X_training,
    y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return clf : the classifier built in this function
    """

    raise NotImplementedError()


def build_NN_classifier(X_training, y_training):
    """
    Build a Nearest Neighbours classifier based on the training set
    X_training, y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return
    clf : the classifier built in this function
    """

    raise NotImplementedError()


def build_SVM_classifier(X_training, y_training):
    """
    Build a Support Vector Machine classifier based on the training set
    X_training, y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return
    clf : the classifier built in this function
    """

    raise NotImplementedError()


if __name__ == "__main__":
    pass
    # call your functions here
    
    prepare_dataset('medical_records.data') # Since medical_records.data is in the same directory as myQuack.py, we can reference it directly
