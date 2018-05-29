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
from sklearn import naive_bayes
from sklearn.cross_validation import train_test_split



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
    tumor_index = 1  # index where classification 'M' or 'B' is located in the data row
    array = []
    
    # iterate reader and append each line from the data set to the empty
    # python array
    for line in reader:
        array.append(line)

    dataset_array = np.array(array)

    # Generate X from the data set by excluding patent ID and tumor classification  
    X = dataset_array[:, 2:]    
    
    array = []  # Empty the helper array to build y
    # for each row in X, get the value in tumor_index, which is the tumor classification
    for i in range(dataset_array.shape[0]):
        tumor_cl = dataset_array[i][tumor_index]
        array.append(tumor_cl)
    
    y = np.array(array)
    
    # Use numpy array built-in indexing to convert 'B' and 'M' to 1 and 0 (see first ML lecture slides)
    y[y == 'M'] = 1
    y[y == 'B'] = 0
    
    # convert char arrays into  arrays i.e. ['1','1','0'...] to [1,1,0]
    X = X.astype(np.float)
    y = y.astype(np.int) 
    
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
    model = naive_bayes.GaussianNB()    # GaussianNB used in classification 

    model.fit(X_training, y_training)
    
    print(model)

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
    ts = 0.33   # this represents the proportion of data to include in the test split, lecture references 1/3 of data 
    
    print(my_team())
    X, y = prepare_dataset('medical_records.data') # Since medical_records.data is in the same directory as myQuack.py, we can reference it directly
    
    # Generate train test statistics
    # see: Naive Bayes Classifier - https://www.youtube.com/watch?v=99MN-rl8jGY
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts)

    build_NB_classifier(X_train, y_train)

