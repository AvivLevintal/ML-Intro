import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import queue
import math

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    classes = data[:,-1]
    sample_size = len(classes)
    unique, counts = np.unique(classes, return_counts=True)
    for i in range(len(unique)):
        gini = gini + (counts[i]/sample_size)**2

    gini = 1 - gini
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    classes = data[:,-1]
    sample_size = len(classes)
    unique, counts = np.unique(classes, return_counts=True)
    for i in range(len(unique)):
        prob = counts[i]/sample_size
        entropy = entropy + prob*np.log2(prob)

    entropy = -entropy
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    column_feature = data[:,feature]
    sample_size = len(data)

    if gain_ratio:
        info_gain = calc_entropy(data)
        split_info = 0.0

        unique = np.unique(column_feature)    

        for value in unique:
            value_matrix = data[column_feature == value]
            value_size = len(value_matrix)
            groups[value] = value_matrix
            info_gain -= (value_size/sample_size)*calc_entropy(value_matrix)
            split_info -= (value_size/sample_size)*np.log2(value_size/sample_size)
        
        
        if split_info == 0.0:
            goodness = 0.0

        else:
            goodness = info_gain/split_info
            

    else:
        
        unique = np.unique(column_feature)
        goodness = impurity_func(data)

        for value in unique:
            value_matrix = data[column_feature == value]
            groups[value] = value_matrix
            value_size = len(value_matrix)
            goodness -= (value_size/sample_size)*impurity_func(value_matrix)

        
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        unique, counts = np.unique(self.data[:,-1], return_counts=True)
        class_values = dict(zip(unique, counts))
        pred = max(class_values, key=class_values.get)

        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values
 
        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
    
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        if self.depth >= self.max_depth:
            self.terminal = True
            return
        
        degree_of_freedom = len(np.unique(self.data[:,self.feature])) - 1
        if self.chi == 1:
            chi_threshold = 0.0
        else:
            chi_threshold = chi_table[degree_of_freedom][self.chi]    
        current_chi = chi_square(self.data, self.feature)

        if current_chi < chi_threshold:
            self.terminal = True
            return

        _ ,b_feature = best_feature(self.data, impurity_func, self.gain_ratio)
        best_col = self.data[:,b_feature]
        unique = np.unique(best_col)
        
    
        for value in unique:
            value_matrix = self.data[best_col == value]
            child = DecisionNode(data=value_matrix, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child, value)

def chi_square(data, feature):
    Df_unique, Df_counts = np.unique(data[:,feature], return_counts=True)
    class_unique, class_counts = np.unique(data[:,-1], return_counts=True)
    class_size = len(data[:,-1])
    chi_final = 0

    for i, value in enumerate(Df_unique):
        Df = Df_counts[i]
        p_y_0 = class_counts[0]/class_size
        p_y_1 = 1 - p_y_0
        e_0 = Df*p_y_0
        e_1 = Df*p_y_1
        pf = np.sum((data[:,-1] == class_unique[0]) & (data[:,feature] == value))
        nf = np.sum((data[:,-1] == class_unique[1]) & (data[:,feature] == value))
        chi_final += (((pf - e_0)**2)/e_0 + ((nf - e_1)**2)/e_1)

    return chi_final


def best_feature(data, impurity_func, gain_ratio=False):
    features = len(data[0]) - 1
    maximal_goodness = 0.0
    b_feature = None
    b_groups = None

    for f in range(features):
        c_goodness, c_groups = goodness_of_split(data, f, impurity_func, gain_ratio=gain_ratio)

        if c_goodness > maximal_goodness:
            maximal_goodness = c_goodness
            b_feature = f
            b_groups = c_groups
        
    
    return b_groups, b_feature

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data=data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    q = queue.Queue()
    q.put(root)
    while not q.empty():

        current = q.get()
        perfect_check = np.unique(current.data[:,-1])
        if len(perfect_check) == 1:
            current.terminal = True
            continue
        _, current.feature = best_feature(current.data, impurity, gain_ratio)
        if current.feature == None:
            current.terminal = True
            continue
        current.split(impurity)
        for child in current.children:
            q.put(child)

    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None

    while not root.terminal == True:
        #If s doesnt exist, meaning if root.feature is not present at children_values
        try:
            child_index = root.children_values.index(instance[root.feature])
            root = root.children[child_index]
        except:
            break
    pred = root.pred

    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0

    sample_size =  dataset.shape[0]
    corret_predictions = 0
    for row in dataset:
        if row[-1] == predict(node, row):
            corret_predictions += 1
    
    accuracy = corret_predictions/sample_size
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        c_tree = build_tree(X_train, calc_entropy, True, max_depth=max_depth)
        training.append(calc_accuracy(c_tree, X_train))
        testing.append(calc_accuracy(c_tree, X_test))

    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []


    for chi_key in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        c_tree = build_tree(X_train, calc_entropy, True, chi=chi_key)
        chi_training_acc.append(calc_accuracy(c_tree, X_train))
        chi_testing_acc.append(calc_accuracy(c_tree, X_test))
        depth.append(get_tree_depth(c_tree))

    return chi_training_acc, chi_testing_acc, depth

def get_tree_depth(node):
    """
    Returns the depth of a decision tree given its root node.
    """
    if node.terminal:
        return node.depth

    depths = []
    for child in node.children:
        depth = get_tree_depth(child)
        depths.append(depth)

    return max(depths)


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = 0
    if node == None:
        return 0
    for child in node.children:
        n_nodes += count_nodes(child)
    n_nodes += 1
    
    return n_nodes






