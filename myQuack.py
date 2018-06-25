import csv
import io
import numpy as np
import pydotplus
import imageio
import matplotlib.pyplot as plt

from sklearn.naive_bayes import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import export_graphviz
from sklearn.model_selection import *
from sklearn.model_selection import cross_val_score
from sklearn import metrics



"""
Scaffolding code for the Machine Learning assignment.
You should complete the provided functions and add more functions and classes
as necessary.
Write a main function that calls the different functions to perform the
required tasks
and repeat your experiments.
"""


# GLOBAL PRIORS
global m_prior
global b_prior


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

    # Use numpy array built-in indexing to convert 'B' and 'M' to 1 and 0
    # (see first ML lecture slides)
    y = np.array(array)
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
    print('B priori', b_prior)
    print('M priori', m_prior)
    clf = GaussianNB(priors=[b_prior, m_prior])   # GaussianNB used in classification
    clf.fit(X_training, y_training)     # Find the best Gaussian fit of the data
    return clf

def build_DT_classifier(X_training, y_training):
    """
    Build a Decision Tree classifier based on the training set X_training,
    y_training.

    @param
    X_training: X_training[i,:] is the ith example
    y_training: y_training[i] is the class label of X_training[i,:]

    @return clf : the classifier built in this function
    """

    #  C R O S S    V A L I D A T I O N

    # Hyper parameter range 2 - 500
    min_split_range = list(range(2, 500))

    # List of cross validation scores
    split_scores = []

    # dictionary
    min_smpl_splt_dict = {}

    # Loop through the hyper parameter range and apply the parameter value to
    # the decision tree classifier
    for split in min_split_range:
        test_tree = DecisionTreeClassifier(min_samples_split=split)
        # perform cross validation using 10 k folds and score on base on
        # accuracy
        scores = cross_val_score(test_tree, X_training, y_training, cv=10, scoring='accuracy')
        # get the mean score from cross val test
        split_scores.append(scores.mean())
        # add the mean split value and mean score to the dictionary
        min_smpl_splt_dict.update({split:scores.mean()})

    #  P L O T T I N G

    # Create a plot to visualize the hyper parameter performance
    plt.plot(min_split_range, split_scores)
    plt.title('Decision Tree - Hyper Parameter Cross Validation')
    plt.xlabel('Split Value for DTree')
    plt.ylabel('Cross Validation Accuracy')
    plt.grid(b=True)
    plt.show()

    # Get optimal number of splits
    optimum_splits = max(min_smpl_splt_dict, key=min_smpl_splt_dict.get)

    # print the standard deviation of validation score
    numpy_scores = np.array(split_scores)
    val_acc_std_dev = numpy_scores.std()
    print("Standard Deviation: ", numpy_scores.std())
    print("Optimal splits: ", optimum_splits)

    # Instantiate decision tree classifier with min_samples_split hyper parameter
    cls = DecisionTreeClassifier(min_samples_split=optimum_splits)

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
    #  C R O S S    V A L I D A T I O N

    # Hyper parameter range 1 - 50
    neighbors_range = list(range(1, 50))

    # List of cross validation scores
    split_scores = []

    # Neighbors dict.
    neighbors_dict = {}

    # Loop through the hyper parameter range and apply the parameter value to
    # the classifier
    for neighbor in neighbors_range:
        test_knn = KNeighborsClassifier(n_neighbors=neighbor)
        # perform cross validation using 10 k folds and score on base on accuracy
        scores = cross_val_score(test_knn, X_training, y_training, cv=10,scoring='accuracy')
        # get the mean score from cross val test
        split_scores.append(scores.mean())
        # add the mean split value and mean score to the dictionary
        neighbors_dict.update({neighbor: scores.mean()})

    # Get highest scoring value from the dictionary
    optimal_neighbors = max(neighbors_dict, key=neighbors_dict.get)
    print('Optimal Neighbors: ', optimal_neighbors)



    #  P L O T T I N G

    # Create a plot to visualize the hyper parameter performance
    plt.plot(neighbors_range, split_scores)
    plt.title('K Nearest Neighbors - Hyper Parameter Cross Validation')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Cross Validation Score - Accuracy')
    plt.grid(b=True)
    plt.show()

    # create instance of knn with optimal parameter
    knn = KNeighborsClassifier(n_neighbors=optimal_neighbors)
    return knn

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
    #  C R O S S    V A L I D A T I O N

    # Hyper parameter Gamma range 1.e-10 - 1.e-2
    gamma_range = np.logspace(-10, -2)

    # List of cross validation scores
    split_scores = []

    # dictionary
    gamam_range_dict = {}

    # Loop through the hyper parameter range and apply the parameter value to
    # the decision tree classifier
    for gamma in gamma_range:
        test_svm = SVC(gamma=gamma)
        # perform cross validation using 10 k folds and score on base on
        # accuracy
        scores = cross_val_score(test_svm, X_training, y_training, cv=10,
                                 scoring='accuracy')
        # get the mean score from cross val test
        split_scores.append(scores.mean())
        # add the mean split value and mean score to the dictionary
        gamam_range_dict.update({gamma: scores.mean()})

    # Create a plot to visualize the hyper parameter performance
    plt.plot(gamma_range, split_scores)
    plt.title('SVM - Hyper Parameter Cross Validation')
    plt.xlabel('Gamma value')
    plt.ylabel('Cross Validation Accuracy')
    plt.grid(b=True)
    plt.show()

    # Get optimal number of splits
    optimal_gamma = max(gamam_range_dict, key=gamam_range_dict.get)
    print('optimal gamma', optimal_gamma)

    svm = SVC(gamma=optimal_gamma, probability=True)

    return svm

    # raise NotImplementedError()

def visualize_tree(dt, path):

    f = io.StringIO()

    export_graphviz(dt, out_file=f)

    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)

    image = imageio.imread(path)

    return image

# Generate  metrics
# @param classifier
# @param classifier name as string
# @param test class prediction variable
def y_pred_class_metrics(classifier, classifier_name, y_pred_class):

    confusion = metrics.confusion_matrix(y_test, y_pred_class)

    print(classifier_name, " - Confusion Matrix  : ", confusion)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    print(classifier_name, ' - classification (testing) accuracy', metrics.accuracy_score(y_test, y_pred_class))
    print(classifier_name, ' - classification error', 1 - metrics.accuracy_score(y_test, y_pred_class))
    print(classifier_name, ' - sensitivity/recall', metrics.recall_score(y_test, y_pred_class))
    print(classifier_name, ' - specificity ', TN / float(TN + FP))
    print(classifier_name, ' - false positive rate ', FP / float(TN + FP))
    print(classifier_name, ' - Precision ', metrics.precision_score(y_test, y_pred_class))

    print(classifier_name,' - cross val score:', cross_val_score(classifier, X, y, cv=10).mean())
    print(classifier_name, ' - cross val error:',
          1- cross_val_score(classifier, X, y, cv=10).mean())
    print(classifier_name,' - training accuracy:', classifier.score(X_train, y_train))
    print(classifier_name, ' - training error:',
          1- classifier.score(X_train, y_train))

# Generate AUC Curve
# @param classifier name as string
# @param test class probability variable
def auc_curve(classifier_name, y_pred_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    title = classifier_name + ' ROC Area Under Curve'
    plt.title(title)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

    print(classifier_name, ' - Area Under Curve',
          metrics.roc_auc_score(y_test, y_pred_prob))

if __name__ == "__main__":

    print(my_team())

    # Generate numpy arrays of data
    X, y = prepare_dataset('medical_records.data')

    # Calculate the priors from the whole dataset based on the proportion of benign and malignant data
    num_mal = np.count_nonzero(y == 1)
    m_prior = num_mal/len(y)
    b_prior = 1 - m_prior
    print("Proportion of Malignant data: %.3f Proportion of Benign: %.3f" % (m_prior, b_prior))

    # Test size 33 %
    ts = .33

    # Split the data into Test and Training Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts)


    # N A I V E    B A Y E S

    # Build the naive bayes classifier
    nb_clf = build_NB_classifier(X_train, y_train)
    nb = nb_clf.fit(X_train, y_train)

    # ADDITIONAL METRICS
    y_pred_class_nb = nb.predict(X_test)
    # Parameters: classifier, name of classifier and prediction results variable
    y_pred_class_metrics(nb_clf, 'Naive Bayes', y_pred_class_nb)

    # ROC AUC DIAGRAM FOR [Decision Tree]
    # prediction probablity variable
    y_pred_prob_nb = nb.predict_proba(X_test)[:, 1]
    # Generate roc auc graph - pass classifier name and prediction probablity variable
    auc_curve('Naive Bayes', y_pred_prob_nb)



    # D E C I S I O N   T R E E

    # instantiate the decision tree classifier
    dt_classifier = build_DT_classifier(X, y)
    # Training the classifier
    dt = dt_classifier.fit(X_train, y_train)
    # Visualize the decision tree
    visualize_tree(dt, 'dt.png')

    # ADDITIONAL METRICS
    y_pred_class_dt = dt.predict(X_test)
    # Parameters: classifier, name of classifier and prediction results variable
    y_pred_class_metrics(dt, 'Decision Tree', y_pred_class_dt)

    # ROC AUC DIAGRAM FOR [Decision Tree]
    # prediction probablity variable
    y_pred_prob_dt = dt.predict_proba(X_test)[:, 1]
    # Generate roc auc graph - pass classifier name and prediction probablity variable
    auc_curve('Decision Tree', y_pred_prob_dt)





    # K   N E A R E S T   N E I G H B O R S

    # instantiate the knn classifier
    knn_classifier = build_NN_classifier(X, y)
    # train the knn classifier
    knn = knn_classifier.fit(X_train, y_train)

    # K-Nearest Neighbors - print training and test scores
    y_pred_class_nn = knn.predict(X_test)
    # Parameters: classifier, name of classifier and prediction results variable
    y_pred_class_metrics(knn, 'Nearest Neighbor', y_pred_class_nn)

    # ROC AUC DIAGRAM FOR [Nearest Neighbor]
    # prediction probablity variable
    y_pred_prob_nn = knn.predict_proba(X_test)[:, 1]
    # Generate roc auc graph - pass classifier name and prediction probablity variable
    auc_curve('Nearest Neighbor', y_pred_prob_nn)




    # S V M    C L A S S I F I E R

    # instantiate the svm classifier
    svm_classifier = build_SVM_classifier(X, y)
    # train the SVM classifier
    svm = svm_classifier.fit(X_train, y_train)

    # SVM SVC  - print training and test scores
    y_pred_class_svm = svm.predict(X_test)
    # Parameters: classifier, name of classifier and prediction results variable
    y_pred_class_metrics(svm, 'Support Vector Machine', y_pred_class_svm)

    # ROC AUC DIAGRAM FOR [Support Vector Machine]
    # prediction probablity variable
    y_pred_prob_svm = svm.predict_proba(X_test)[:, 1]
    # Generate roc auc graph - pass classifier name and prediction probablity variable
    auc_curve('Support Vector Machine', y_pred_prob_svm)
