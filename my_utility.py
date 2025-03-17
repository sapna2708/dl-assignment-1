import numpy as np

def Xavier_initializer(dim):
    '''
    Xavier weight initialization for neural networks.

    Parameters:
    - dim: Tuple (output_dim, input_dim) representing the dimensions of the weight matrix.

    Returns:
    - A numpy array of shape (output_dim, input_dim) with Xavier-initialized values.
    '''
    xavier_stddev = np.sqrt(2 / (dim[1] + dim[0]))
    return np.random.normal(0, xavier_stddev, size=(dim[0], dim[1]))

def random_initializer(dim):
    '''
    Random weight initialization for neural networks.

    Parameters:
    - dim: Tuple (output_dim, input_dim) representing the dimensions of the weight matrix.

    Returns:
    - A numpy array of shape (output_dim, input_dim) with randomly initialized values.
    '''
    return np.random.normal(0, 1, size=(dim[0], dim[1]))

def He_initializer(dim):
    '''
    He weight initialization for neural networks.

    Parameters:
    - dim: Tuple (output_dim, input_dim) representing the dimensions of the weight matrix.

    Returns:
    - A numpy array of shape (output_dim, input_dim) with He-initialized values.
    '''
    He_stddev = np.sqrt(2 / (dim[1]))
    return np.random.normal(0, 1, size=(dim[0], dim[1])) * He_stddev


def meanSquaredErrorLoss(Y_true, Y_pred):
    '''
    Calculates the Mean Squared Error (MSE) loss between true and predicted values.

    Arguments:
    - Y_true (numpy.ndarray): True output labels.
    - Y_pred (numpy.ndarray): Predicted output labels.

    Returns:
    - float: Mean Squared Error loss.
    '''
    return np.mean((Y_true - Y_pred) * (Y_true - Y_pred))

def crossEntropyLoss( Y_true, Y_pred):
    '''
    Calculates the Cross-Entropy loss between true and predicted probability distributions.

    Arguments:
    - Y_true (numpy.ndarray): True output labels in one-hot encoded form.
    - Y_pred (numpy.ndarray): Predicted probability distributions.

    Returns:
    - float: Cross-Entropy loss.
    '''
    eps = 1e-15
    Y_pred = np.clip(Y_pred,eps,1.0-eps)
    loss = -np.sum(Y_true*np.log(Y_pred),axis=1)
    loss = np.mean(loss)
    return loss

def oneHotEncoding(num_classes, train_raw):
    '''
    Performs one-hot encoding on the provided labels.

    Parameters:
    - Y_train_raw (numpy.ndarray): Raw output labels.

    Returns:
    - Ydata (numpy.ndarray): One-hot encoded representation of the labels.
    '''
    return np.array([[1.0 if int(train_raw[i]) == j else 0.0 for i in range(train_raw.shape[0])] for j in range(num_classes)])

def printAccuracy(epoch, training_loss, training_acc, validation_acc, elapsed, alpha):
    print(f"Epoch: {epoch}, "
          f"Loss: {training_loss:.3e}, "
          f"Training Accuracy: {training_acc:.2f}, "
          f"Validation Accuracy: {validation_acc:.2f}, "
          f"Time: {elapsed:.2f}s, "
          f"Learning Rate: {alpha:.3e}")


def accuracy(Y_true, Y_pred, data_size):
    """
    Calculates the accuracy of the model's predictions.

    Arguments:
    - Y_true (numpy.ndarray): True output labels in one-hot encoded form.
    - Y_pred (numpy.ndarray): Predicted output labels in one-hot encoded form.
    - data_size (int): Number of samples in the dataset.

    Returns:
    - float: Accuracy of the model.
    - list: True labels.
    - list: Predicted labels.
    """
    Y_true_vals = np.argmax(Y_true, axis=0).tolist()
    Y_pred_vals = np.argmax(Y_pred, axis=0).tolist()
    
    correct_vals = sum(yt == yp for yt, yp in zip(Y_true_vals, Y_pred_vals))
    acc = correct_vals / data_size
    
    return acc, Y_true_vals, Y_pred_vals


def sigmoid(z):
    # z = np.clip(z,500,-500)
    return 1.0 / (1 + np.exp(-(z)))


def tanh(z):
    return np.tanh(z)


def sin(z):
    return np.sin(z)


def reLu(z):
    return (z>0)*(z) + ((z<0)*(z)*0.01)
    #return np.maximum(z,0)
    #return np.where(z<0, 0.01*z, z)

def softmax(Z):
    # Z = np.clip(Z,500,-500)
    Z -= np.max(Z)
    # Compute softmax
    exp_Z = np.exp(Z)
    softmax_output = exp_Z / np.sum(exp_Z)
    return softmax_output


def del_sigmoid(z):
    # z = np.clip(z,500,-500)
    return  (1.0 / (1 + np.exp(-(z))))*(1 -  1.0 / (1 + np.exp(-(z))))

def del_tanh(z):
    return 1 - np.tanh(z) ** 2


def del_reLu(z):
    return (z>0)*np.ones(z.shape) + (z<0)*(0.01*np.ones(z.shape) )
