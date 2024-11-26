import numpy as np

def calculate_priors(y_train, classes):
    # counts of each classes in the training set
    counts = np.bincount(y_train, minlength=len(classes))
    
    # P( Y = y_k )
    priors = counts / y_train.shape[0]
    
    # log( P( Y = y_k ) )
    log_priors = np.log(priors)

    return log_priors

def calculate_conditionals(x_train, y_train, classes, alpha = 0):
    # a one hot array of size (samples, 3) indicating the class of each sample
    y_train_onehot = np.zeros((x_train.shape[0], len(classes)))
    y_train_onehot[np.arange(x_train.shape[0]), y_train] = 1
    
    # T[ i, j ] is number of occurences of w_j in class y_i
    T = y_train_onehot.T @ x_train
    
    # T_sum[i] is the total number of words in class y_i
    T_sum = T.sum(axis=1)
    
    # conditionals[ i, j ] = P( X_j | Y = y_i )
    conditionals = (T + alpha) / (T_sum[:, np.newaxis] + alpha * x_train.shape[1])
    conditionals = np.clip(conditionals, a_min=1e-12, a_max=None)
    
    # log_conditionals[ i, j ] = log( P( X_j | Y = Y_i ) )
    log_conditionals = np.log(conditionals)

    return log_conditionals


def predict(test_data, log_priors, log_conditionals):
    return np.argmax(log_priors[np.newaxis, :] + test_data @ log_conditionals.T, axis=1)

def confusion_matrix(labels, predicted_labels, size=3):
    cm = np.zeros((size,size))
    for i in range(len(labels)):
        cm[labels[i]][predicted_labels[i]] += 1
    return cm

# --------------------------------------------------------------------------------------

def multinomial_nb(TRAINING_FILES, TEST_FILES, classes, alpha):
    x_train = np.genfromtxt(TRAINING_FILES[0], delimiter=",", dtype=str)
    y_train = np.genfromtxt(TRAINING_FILES[1], delimiter=",", dtype=int)
    x_train = x_train[1:].astype(int)

    x_test = np.genfromtxt(TEST_FILES[0], delimiter=",", dtype=str)
    y_test = np.genfromtxt(TEST_FILES[1], delimiter=",", dtype=int)
    x_test = x_test[1:].astype(int)

    priors = calculate_priors(y_train, classes)
    conditionals = calculate_conditionals(x_train, y_train, classes, alpha=alpha)

    predicted_labels = predict(x_test, priors, conditionals)

    cm = confusion_matrix(y_test, predicted_labels)
    accuracy = np.trace(cm) / cm.sum()

    return accuracy,cm

