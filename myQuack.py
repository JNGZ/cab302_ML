import csv
import io
import numpy as np
import pydotplus
import imageio
import matplotlib.pyplot as plt

from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import *
from sklearn.model_selection import cross_val_score


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

    data_input = open(dataset_path, 'rt')
    reader = csv.reader(data_input)
    # index where classification 'M' or 'B' is located in the data row
    tumor_index = 1
    array = []

    # iterate reader and append each line from the data set to the empty
    # python array
    for line in reader:
        array.append(line)

    dataset_array = np.array(array)

    # Generate X from the data set by excluding patent ID
    # and tumor classification
    X = dataset_array[:, 2:]

    array = []  # Empty the helper array to build y

    # for each row in X, get the value in tumor_index,
    # which is the tumor classification
    for i in range(dataset_array.shape[0]):
        tumor_cl = dataset_array[i][tumor_index]
        array.append(tumor_cl)

    y = np.array(array)

    # Use numpy array built-in indexing to convert 'B' and 'M' to 1 and 0
    # (see first ML lecture slides)
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


def build_DT_classifier(X, y):
    """
    Build a Decision Tree classifier based on the training set X_training,
    y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return clf : the classifier built in this function
    """

    #  C R O S S    V A L I D A T I O N
    # Hyper parameter range 2 - 100
    min_split_range = list(range(2,500))

    # List of cross validation scores
    split_scores = []

    # Loop through the hyper parameter range and apply the parameter value to
    # the decision tree classifier
    # then use the cross_vale_score function to split the data into 10 folds
    # for cross validation
    # then return the accuracy score for the 10 fold validation
    # finally, append the mean of the 10 scores to the split_scores list
    for split in (min_split_range):
        test_tree = DecisionTreeClassifier(min_samples_split=split)
        scores = cross_val_score(test_tree, X, y, cv=10, scoring='accuracy')
        split_scores.append(scores.mean())

    # Create a plot to visualize the hyper parameter performance
    plt.plot(min_split_range, split_scores)
    plt.xlabel('Split Value for DTree')
    plt.ylabel('Cross Validation Accuracy')
    plt.grid(b=True)
    plt.show()

    # Instantiate decision tree classifier with min_samples_split hyper parameter
    cls = DecisionTreeClassifier(min_samples_split=190)

    return cls


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


def visualize_tree(dt, path):

    f = io.StringIO()

    export_graphviz(dt, out_file=f)

    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)

    image = imageio.imread(path)

    return image


if __name__ == "__main__":
    
    print(my_team())

    # Prepare and format the raw data
    X, y = prepare_dataset('medical_records.data')

    # Test size 33 %
    ts = .33

    # Split the data into Test and Training Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts)

    # instantiate the decision tree classifier
    dt_classifier = build_DT_classifier(X, y)
    # Training the classifier
    dt = dt_classifier.fit(X_train, y_train)
    # Visualize the decision tree
    visualize_tree(dt, 'dt.png')

    # Decision tree cross validation score
    validation_scores = cross_val_score(dt, X, y, cv=10, scoring='accuracy')
    val_score_mean = validation_scores.mean()
    print('Validation accuracy: ', val_score_mean)

    # Get standard deviation of validation accuracy
    numpy_scores = np.array(validation_scores)
    val_acc_std_dev = numpy_scores.std()
    print('Standard deviation of validation using 190 hyper parameter', val_acc_std_dev)

    # Decision tree prediction on training set
    dt_pred = dt.predict(X_train)
    # Accuracy score for prediction
    prediction_score = accuracy_score(y_train, dt_pred)
    print('Training accuracy: ', prediction_score)

    # Decision tree prediction on test set
    dt_pred = dt.predict(X_test)
    # Accuracy score for prediction
    prediction_score = accuracy_score(y_test, dt_pred)
    print('Test accuracy: ', prediction_score)



