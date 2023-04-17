###### Your ID ######
# ID1: 315111575
# ID2: 207496191
#####################

# imports 
import numpy as np
import pandas as pd
from itertools import permutations

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (y - np.mean(y)) / (np.max(y) - np.min(y))


    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    one_matrix = np.ones(X.shape[0])
    X = np.column_stack((one_matrix, X))
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
        - J: the cost associated with the current set of parameters (single number).
    """
    
    h = np.inner(theta,X)
    J = np.inner(h-y,h-y)/(2*len(y))
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """ 
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy() # optional: theta outside the function will not change

    J_history = []
    m = len(y)

    h = np.inner(X,theta)

    for i in range(0, num_iters):
        J_history.append(compute_cost(X, y, theta))
        sigma = np.zeros(len(theta))
        h = np.dot(X, theta) - y
        sigma = np.dot(X.transpose(), h) / m
        theta = theta - alpha*sigma



    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)), np.transpose(X))
    pinv_theta = np.matmul(pinv, y)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change

    J_history = []
    m = len(y)

    h = np.inner(X,theta)

    for i in range(0, num_iters):
        J_history.append(compute_cost(X, y, theta))
        if i > 0 and (J_history[i-1] - J_history[i] < 10**-8):
            break
        sigma = np.zeros(len(theta))
        h = np.dot(X, theta) - y
        sigma = np.dot(X.transpose(), h) / m
        theta = theta - alpha*sigma



    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} 
    np.random.seed(42)
    theta = np.random.random(size=len(X_train.transpose()))

    for alpha in alphas:
        o_theta, _ = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, o_theta)
     
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.
    
    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    not_selected_features = np.arange(0, len(X_train[0,:]), 1, dtype=int)
    general_features = np.arange(0, len(X_train[0,:]), 1, dtype=int)
    performance = []
    feature_matrix = [] 

    feature_matrix.append(np.ones(X_train.shape[0]).transpose())
    feature_matrix.append(np.ones(X_val.shape[0]).transpose())
    
    for i in range(0,5):
        theta = np.random.random(size=i + 2)
        for j in range(0, len(not_selected_features)):
        
            feature_matrix[0] = np.column_stack((feature_matrix[0], X_train[:,j]))
            feature_matrix[1] = np.column_stack((feature_matrix[1], X_val[:,j]))
            
            o_theta, _ = efficient_gradient_descent(feature_matrix[0], y_train, theta, best_alpha, iterations)    
            performance.append(compute_cost(feature_matrix[1], y_val, o_theta))
            
            for c in range(0,2):
                feature_matrix[c] = np.delete(feature_matrix[c], i + 1, 1)
 
        cur_min = np.argmin(performance)
        selected_features.append(general_features[cur_min])
        feature_matrix[0] = np.column_stack((feature_matrix[0], X_train[:,cur_min]))
        feature_matrix[1] = np.column_stack((feature_matrix[1], X_val[:,cur_min]))
        not_selected_features = np.delete(not_selected_features, cur_min)
        performance = []

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:  
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    new_cols = {}
    for j,feature in enumerate(df):
        for i in range (j, len(df.columns)):
            if i == j:
                col_name = feature + '^2'
            else:
                col_name = feature + '*' + df.columns[i]
            new_cols[col_name] = df_poly[df_poly.columns[j]] * df_poly[df_poly.columns[i]]
    df_poly = pd.concat([df_poly,pd.DataFrame(new_cols)], axis=1)
    return df_poly

