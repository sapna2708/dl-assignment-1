import time
import numpy as np
from my_utility import accuracy,oneHotEncoding,random_initializer
from my_utility import sigmoid,tanh,reLu,del_sigmoid,del_reLu,del_tanh,softmax
from my_utility import Xavier_initializer,random_initializer,He_initializer
from my_utility import oneHotEncoding,accuracy,printAccuracy
from my_utility import crossEntropyLoss,meanSquaredErrorLoss
from constant import SIGMOID_KEY,TANH_KEY,RELU_KEY
from constant import XAVIER_KEY,RANDOM_KEY,HE_KEY
from constant import SGD_KEY,MGD_KEY,NAG_KEY,RMSPROP_KEY,ADAM_KEY,NADAM_KEY
from constant import CROSS_ENTROPY_KEY,MEAN_SQUARE_KEY
from constant import GRAD_A,GRAD_W,GRAD_H,GRAD_B

class FeedForwardNeuralNetwork:
    '''
    Neural network model for feedforward architecture.

    Attributes:
    - hidden_layers (List[int]): List representing the number of neurons in each hidden layer.
    - output_layer_neuron (int): Number of neurons in the output layer.
    - X_train_raw (numpy.ndarray): Raw training input data.
    - Y_train_raw (numpy.ndarray): Raw training output labels.
    - N_train (int): Number of training samples.
    - X_val_raw (numpy.ndarray): Raw validation input data.
    - Y_val_raw (numpy.ndarray): Raw validation output labels.
    - N_val (int): Number of validation samples.
    - X_test_raw (numpy.ndarray): Raw test input data.
    - Y_test_raw (numpy.ndarray): Raw test output labels.
    - N_test (int): Number of test samples.
    - batch_size (int): Size of the mini-batch used during training.
    - weight_decay (float): Weight decay regularization parameter.
    - learning_rate (float): Learning rate for optimization.
    - epochs (int): Number of training epochs.
    - activation_fun (str): Activation function used in hidden layers.
    - initializer (str): Weight initialization method - "RANDOM" (default), "XAVIER", or "HE".
    - optimizer (str): Optimization algorithm - "SGD" (default), "MBGD", "NAGD", "RMS", "ADAM", or "NADAM".
    - loss_function (str): Loss function used for training - "CROSS_ENTROPY" (default) or MEAN_SQUARE_KEY.

    Methods:
    - __init__: Initializes the neural network with the provided parameters and initializes weights and biases.
    - initializeNeuralNet: Helper function to initialize weights and biases for the neural network layers.

    Note:
    - The network architecture is defined by the combination of hidden_layers and output_layer_neuron.
    - The input data is expected to be flattened, with dimensions (num_features, num_samples).
    - Raw input data is normalized to the range [0, 1].
    - The activation function and its derivative are specified based on the chosen activation_fun.
    - The initializer for weights is selected from "RANDOM" (default), "XAVIER", or "HE".
    - The optimization algorithm can be chosen from "SGD" (default), "MBGD", "NAGD", "RMS", "ADAM", or "NADAM".
    - The loss function for training is chosen from "CROSS_ENTROPY" (default) or MEAN_SQUARE_KEY.
    '''
    def __init__(
        self,
        num_hidden_layers,
        num_hidden_neurons,
        X_train_raw,
        Y_train_raw,
        N_train,
        X_val_raw,
        Y_val_raw,
        N_val,
        X_test_raw,
        Y_test_raw,
        N_test,
        optimizer,
        batch_size,
        weight_decay,
        learning_rate,
        max_epochs,
        activation,
        initializer,
        loss

    ):

        img_width = 255
        self.num_classes = np.max(Y_train_raw) # NUM_CLASSES
        self.num_classes += 1

        self.num_hidden_layers = num_hidden_layers
        print()
        self.num_hidden_neurons = num_hidden_neurons
        print()
        self.output_layer_size = self.num_classes

        self.img_wid = X_train_raw.shape[2]

        self.img_hei = X_train_raw.shape[1]

        self.img_flat_size = self.img_hei * self.img_wid

        # self.network = layers
        self.network = (
            [self.img_flat_size]
            + num_hidden_layers * [num_hidden_neurons]
            + [self.output_layer_size]
        )

        self.N_tr = N_train
        self.NVal = N_val
        self.N_te = N_test

        x_raw_param = X_train_raw.reshape(
                X_train_raw.shape[0], X_train_raw.shape[1] * X_train_raw.shape[2]
            )

        self.X_tr = np.transpose(x_raw_param)
        x_raw_param = X_test_raw.reshape(
                X_test_raw.shape[0], X_test_raw.shape[1] * X_test_raw.shape[2]
            )

        self.X_te = np.transpose(x_raw_param)
        x_raw_param = X_val_raw.reshape(
                X_val_raw.shape[0], X_val_raw.shape[1] * X_val_raw.shape[2]
            )

        self.XVal = np.transpose(x_raw_param)


        self.X_te = self.X_te / img_width
        self.X_tr = self.X_tr / img_width
        self.XVal = self.XVal / img_width

        self.Y_train = oneHotEncoding(self.num_classes,Y_train_raw)
        self.Y_val = oneHotEncoding(self.num_classes,Y_val_raw)
        self.Y_test = oneHotEncoding(self.num_classes,Y_test_raw)


        self.Activations_dict = dict([
            (SIGMOID_KEY, sigmoid),
            (TANH_KEY, tanh),
            (RELU_KEY, reLu)
            ])
        self.DerActivation_dict = dict([
            (SIGMOID_KEY, del_sigmoid),
            (TANH_KEY, del_tanh),
            (RELU_KEY, del_reLu)
        ])

        self.Optimizer_dict = dict([
            (SGD_KEY, self.sgdOptimizer),
            (MGD_KEY, self.mgdOptimizer),
            (NAG_KEY, self.nagOptimizer),
            (RMSPROP_KEY, self.rmsOptimizer),
            (ADAM_KEY, self.adamOptimizer),
            (NADAM_KEY, self.nadamOptimizer)
        ])

        self.Initializer_dict = dict([
            (XAVIER_KEY, Xavier_initializer),
            (RANDOM_KEY, random_initializer),
            (HE_KEY, He_initializer)
        ])


        self.optimizer = self.Optimizer_dict[optimizer]

        self.activation = self.Activations_dict[activation]

        self.der_activation = self.DerActivation_dict[activation]
        print(self.optimizer)
        self.loss_function = loss

        self.initializer = self.Initializer_dict[initializer]

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        print(max_epochs)

        self.alpha = learning_rate

        init_results = self.initializeNeuralNet(self.network)
        self.weights = init_results[0]
        self.biases = init_results[1]

    def L2RegLoss(self, weight_decay):
        '''
        Calculates the L2 regularization loss for the neural network weights.

        Arguments:
        - weight_decay (float): Regularization parameter.

        Returns:
        - float: L2 regularization loss.
        '''
        ALPHA = weight_decay
        total_norm = 0

        for i in range(len(self.weights)):
            total_norm += np.linalg.norm(self.weights[str(i + 1)]) ** 2

        return ALPHA * total_norm

    def predict(self,X,length_dataset):
        '''
        Generates predictions for a given input dataset.

        Arguments:
        - X (numpy.ndarray): Input dataset.
        - length_dataset (int): Number of samples in the dataset.

        Returns:
        - numpy.ndarray: Predicted output matrix.
        '''
        Y_pred = []

        for i in range(length_dataset):
            Y, H, A = self.forwardPropagate(
                X[:, i].reshape(self.img_flat_size, 1),
                self.weights,
                self.biases,
            )
            Y_pred.append(Y.reshape(self.num_classes,))

        return np.array(Y_pred).T

    def initializeNeuralNet(self, layers):
        '''
        Initializes weights and biases for the neural network layers.

        Parameters:
        - layers (List[int]): List representing the number of neurons in each layer.

        Returns:
        - weights (dict): Dictionary containing weight matrices for each layer.
        - biases (dict): Dictionary containing bias vectors for each layer.
        '''
        weights, biases = {}, {}

        for l in range(len(layers) - 1):
            key = str(l + 1)
            weights[key] = self.initializer(dim=[layers[l + 1], layers[l]])
            biases[key] = np.zeros((layers[l + 1], 1))

        return weights, biases

    def initWeight(self):
        result = []
        for l in range(len(self.network) - 1):
            result.append(np.zeros((self.network[l + 1], self.network[l])))
        return result

    def initBias(self):
        result = []
        for l in range(len(self.network) - 1):
            result.append(np.zeros((self.network[l + 1], 1)))
        return result
    def forwardPropagate(self, X_train_batch, weights, biases):
        """
        Performs forward propagation to calculate the output of the neural network.

        Arguments:
        - X_train_batch (numpy.ndarray): Input matrix for a batch of training data.
        - weights (dict): Dictionary containing weight matrices for each layer.
        - biases (dict): Dictionary containing bias vectors for each layer.

        Returns:
        - Y_cap (numpy.ndarray): Predicted output matrix for the given input batch.
        - H (dict): Dictionary containing activation values for each layer during forward propagation.
        - A (dict): Dictionary containing preactivation values for each layer during forward propagation.
        """
        num_layers = len(weights) + 1  # Total number of layers
        H, A = {"0": X_train_batch}, {"0": X_train_batch}  # Initialize activations and preactivations

        # Forward propagation for hidden layers
        for l in range(1, num_layers - 1):
            key = str(l)
            W, b = weights[key], biases[key]
            A[key] = np.matmul(W, H[str(l - 1)]) + b
            H[key] = self.activation(A[key])

        # Last layer (output layer, no activation for regression)
        last_key = str(num_layers - 1)
        W, b = weights[last_key], biases[last_key]
        A[last_key] = np.matmul(W, H[str(num_layers - 2)]) + b
        Y_cap = softmax(A[last_key])  # Apply softmax to final layer

        H[last_key] = Y_cap
        return Y_cap, H, A
    def backPropagate(self, Y, H, A, Y_train_batch, weight_decay=0):
        """
        Backpropagate through the neural network to compute gradients with respect to weights and biases.

        Parameters:
        - Y: The predicted output of the neural network.
        - H: Dictionary containing hidden layer outputs.
        - A: Dictionary containing pre-activation values for each layer.
        - Y_train_batch: The true output labels.
        - weight_decay: Regularization parameter to control overfitting (default is 0).

        Returns:
        - gradients_weights: List of weight gradients for each layer.
        - gradients_biases: List of bias gradients for each layer.
        """

        num_layers = len(self.network)
        gradients_weights, gradients_biases = [], {}

        # Compute initial gradient at output layer
        grad_A = {}
        if self.loss_function == CROSS_ENTROPY_KEY:
            grad_A[str(num_layers - 1)] = -(Y_train_batch - Y)
        elif self.loss_function == MEAN_SQUARE_KEY:
            grad_A[str(num_layers - 1)] = 2 * (Y - Y_train_batch) * Y * (1 - Y)

        # Backpropagation loop
        for l in range(num_layers - 2, -1, -1):
            layer_key = str(l + 1)
            prev_layer_key = str(l)

            # Compute weight gradients
            grad_W = np.outer(grad_A[layer_key], H[prev_layer_key])
            if weight_decay != 0:
                grad_W += weight_decay * self.weights[layer_key]

            # Compute bias gradients
            grad_B = grad_A[layer_key]

            # Store gradients
            gradients_weights.append(grad_W)
            gradients_biases[layer_key] = grad_B

            # Compute activation gradients for the previous layer
            if l > 0:
                grad_H = np.matmul(self.weights[layer_key].T, grad_A[layer_key])
                grad_A[prev_layer_key] = grad_H * self.der_activation(A[prev_layer_key])
            else:
                grad_H = np.matmul(self.weights[layer_key].T, grad_A[layer_key])
                grad_A[prev_layer_key] = grad_H * A[prev_layer_key]  # No activation function applied

        return gradients_weights, list(gradients_biases.values())

    def sgd(self, epochs, length_dataset, learning_rate, weight_decay=0):
        """
        Implement Stochastic Gradient Descent (SGD) optimization for training the neural network.

        Parameters:
        - epochs: Number of training epochs.
        - length_dataset: Number of samples in the training dataset.
        - learning_rate: Learning rate for the optimization.
        - weight_decay: Regularization parameter to control overfitting (default is 0).

        Returns:
        - train_loss: List of training losses per epoch.
        - train_accu: List of training accuracies per epoch.
        - val_accu: List of validation accuracies per epoch.
        - Y_pred: Predicted outputs after training.
        """
        train_loss, train_accu, val_accu = [], [], []
        network_size = len(self.network)

        # Extract and reshape training data
        X_train, Y_train = self.X_tr[:, :length_dataset], self.Y_train[:, :length_dataset]
        X_train = X_train.reshape(self.img_flat_size, length_dataset)
        Y_train = Y_train.reshape(self.num_classes, length_dataset)

        for epoch in range(epochs):
            start_time = time.time()
            LOSS = []

            # Initialize weight and bias gradients
            del_w, del_b = self.initWeight(), self.initBias()

            for i in range(length_dataset):
                # Forward pass
                Y_cap, H, A = self.forwardPropagate(X_train[:, i].reshape(self.img_flat_size, 1), self.weights, self.biases)

                # Backward pass
                grad_weights, grad_biases = self.backPropagate(Y_cap, H, A, Y_train[:, i].reshape(self.num_classes, 1))

                del_w = grad_weights[::-1]  # Reverse order
                del_b = grad_biases[::-1]

                # Compute loss with L2 regularization
                l2_loss = self.L2RegLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(Y_train[:, i].reshape(self.num_classes, 1), Y_cap) + l2_loss)
                else:
                    LOSS.append(crossEntropyLoss(Y_train[:, i].reshape(self.num_classes, 1), Y_cap) + l2_loss)

                # Update weights and biases
                for j in range(len(self.weights)):
                    self.weights[str(j + 1)] -= learning_rate * del_w[j]

                for j in range(len(self.biases)):
                    self.biases[str(j + 1)] -= learning_rate * del_b[j]

            elapsed_time = time.time() - start_time

            # Compute training accuracy
            Y_pred = self.predict(self.X_tr, self.N_tr)
            train_loss.append(np.mean(LOSS))
            train_accu.append(accuracy(Y_train, Y_pred, length_dataset)[0])

            # Compute validation accuracy
            Y_val_pred = self.predict(self.XVal, self.NVal)
            val_accu.append(accuracy(self.Y_val, Y_val_pred, self.NVal)[0])

            # Compute validation loss
            l2_reg = self.L2RegLoss(weight_decay)
            val_loss = np.mean(
                meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T) + l2_reg
                if self.loss_function == MEAN_SQUARE_KEY
                else crossEntropyLoss(self.Y_val.T, Y_val_pred.T) + l2_reg
            )

            printAccuracy(epoch, train_loss[-1], train_accu[-1], val_accu[-1], elapsed_time, self.alpha)

        return train_loss, train_accu, val_accu, Y_pred

    def sgdOptimizer(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        """
        Train the neural network using Stochastic Gradient Descent (SGD) with Mini-Batch updates.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - train_loss (list): List of training losses for each epoch.
        - train_acc (list): List of training accuracies for each epoch.
        - val_accu (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.
        """
        X_train, Y_train = self.X_tr[:, :length_dataset], self.Y_train[:, :length_dataset]

        train_loss, train_acc, val_accu = [], [], []
        num_layers = len(self.network)

        for epoch in range(epochs):
            start_time = time.time()

            # Shuffle dataset
            indices = np.random.permutation(length_dataset)
            X_train, Y_train = X_train[:, indices], Y_train[:, indices]

            LOSS = []
            deltaw, deltab = self.initWeight(), self.initBias()
            num_points_seen = 0

            for i in range(length_dataset):
                # Forward and backward propagation
                Y, H, A = self.forwardPropagate(X_train[:, i].reshape(self.img_flat_size, 1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[:, i].reshape(self.num_classes, 1))

                # Accumulate gradients
                for j in range(num_layers - 1):
                    deltaw[j] += grad_weights[num_layers - 2 - j]
                    deltab[j] += grad_biases[num_layers - 2 - j]

                # Compute loss with L2 regularization
                l2_loss = self.L2RegLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(Y_train[:, i].reshape(self.num_classes, 1), Y) + l2_loss)
                else:
                    LOSS.append(crossEntropyLoss(Y_train[:, i].reshape(self.num_classes, 1), Y) + l2_loss)

                num_points_seen += 1

                # Update weights and biases at batch interval
                if num_points_seen % batch_size == 0:
                    self.weights = {str(j + 1): (self.weights[str(j + 1)] - learning_rate * deltaw[j] / batch_size)
                                    for j in range(len(self.weights))}
                    self.biases = {str(j + 1): (self.biases[str(j + 1)] - learning_rate * deltab[j] / batch_size)
                                for j in range(len(self.biases))}

                    # Reset gradients
                    deltaw, deltab = self.initWeight(), self.initBias()

            elapsed_time = time.time() - start_time

            # Compute training and validation metrics
            Y_pred = self.predict(self.X_tr, self.N_tr)
            train_loss.append(np.mean(LOSS))
            train_acc.append(accuracy(Y_train, Y_pred, length_dataset)[0])

            Y_val_pred = self.predict(self.XVal, self.NVal)
            val_accu.append(accuracy(self.Y_val, Y_val_pred, self.NVal)[0])

            # Compute validation loss
            l2_reg = self.L2RegLoss(weight_decay)
            val_loss = np.mean(meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T) + l2_reg
                            if self.loss_function == MEAN_SQUARE_KEY
                            else crossEntropyLoss(self.Y_val.T, Y_val_pred.T) + l2_reg)

            printAccuracy(epoch, train_loss[-1], train_acc[-1], val_accu[-1], elapsed_time, self.alpha)

        return train_loss, train_acc, val_accu, Y_pred

    def mgdOptimizer(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        """
        Train the neural network using the Mini-Batch Gradient Descent (MGD) optimization algorithm with momentum.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - train_loss (list): List of training losses for each epoch.
        - train_acc (list): List of training accuracies for each epoch.
        - val_accu (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.
        """

        GAMMA = 0.9  # Momentum factor

        X_train, Y_train = self.X_tr[:, :length_dataset], self.Y_train[:, :length_dataset]

        train_loss, train_acc, val_accu = [], [], []
        num_layers = len(self.network)

        # Initialize previous velocity values for momentum
        prev_v_w, prev_v_b = self.initWeight(), self.initBias()

        for epoch in range(epochs):
            start_time = time.time()

            # Shuffle dataset
            indices = np.random.permutation(length_dataset)
            X_train, Y_train = X_train[:, indices], Y_train[:, indices]

            LOSS = []
            deltaw, deltab = self.initWeight(), self.initBias()
            num_points_seen = 0

            for i in range(length_dataset):
                # Forward and backward propagation
                Y, H, A = self.forwardPropagate(X_train[:, i].reshape(self.img_flat_size, 1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[:, i].reshape(self.num_classes, 1))

                # Accumulate gradients
                for j in range(num_layers - 1):
                    deltaw[j] += grad_weights[num_layers - 2 - j]
                    deltab[j] += grad_biases[num_layers - 2 - j]

                # Compute loss with L2 regularization
                l2_loss = self.L2RegLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(Y_train[:, i].reshape(self.num_classes, 1), Y) + l2_loss)
                else:
                    LOSS.append(crossEntropyLoss(Y_train[:, i].reshape(self.num_classes, 1), Y) + l2_loss)

                num_points_seen += 1

                # Update weights and biases at batch interval with momentum
                if num_points_seen % batch_size == 0:
                    v_w = [GAMMA * prev_v_w[j] + learning_rate * deltaw[j] / batch_size for j in range(num_layers - 1)]
                    v_b = [GAMMA * prev_v_b[j] + learning_rate * deltab[j] / batch_size for j in range(num_layers - 1)]

                    # Update weights and biases
                    self.weights = {str(j + 1): self.weights[str(j + 1)] - v_w[j] for j in range(len(self.weights))}
                    self.biases = {str(j + 1): self.biases[str(j + 1)] - v_b[j] for j in range(len(self.biases))}

                    # Store previous velocity
                    prev_v_w, prev_v_b = v_w, v_b

                    # Reset accumulated gradients
                    deltaw, deltab = self.initWeight(), self.initBias()

            elapsed_time = time.time() - start_time

            # Compute training and validation metrics
            Y_pred = self.predict(self.X_tr, self.N_tr)
            train_loss.append(np.mean(LOSS))
            train_acc.append(accuracy(Y_train, Y_pred, length_dataset)[0])

            Y_val_pred = self.predict(self.XVal, self.NVal)
            val_accu.append(accuracy(self.Y_val, Y_val_pred, self.NVal)[0])

            # Compute validation loss
            l2_reg = self.L2RegLoss(weight_decay)
            val_loss = np.mean(meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T) + l2_reg
                            if self.loss_function == MEAN_SQUARE_KEY
                            else crossEntropyLoss(self.Y_val.T, Y_val_pred.T) + l2_reg)

            printAccuracy(epoch, train_loss[-1], train_acc[-1], val_accu[-1], elapsed_time, self.alpha)

        return train_loss, train_acc, val_accu, Y_pred

    def nagOptimizer(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        """
        Train the neural network using the Nesterov Accelerated Gradient (NAG) optimization algorithm.
        """
        GAMMA = 0.9
        X_train, Y_train = self.X_tr[:, :length_dataset], self.Y_train[:, :length_dataset]

        train_loss, train_acc, val_accu = [], [], []
        num_layers = len(self.network)
        prev_v_w, prev_v_b = self.initWeight(), self.initBias()
        num_points_seen, epoch = 0, 0

        while epoch < epochs:
            start_time = time.time()
            offset = np.random.permutation(length_dataset)
            X_train, Y_train = X_train[:, offset], Y_train[:, offset]

            LOSS, deltaw, deltab = [], self.initWeight(), self.initBias()
            v_w = [GAMMA * prev_v_w[l] for l in range(num_layers - 1)]
            v_b = [GAMMA * prev_v_b[l] for l in range(num_layers - 1)]
            
            for i in range(length_dataset):
                winter = {str(l + 1): self.weights[str(l + 1)] - v_w[l] for l in range(num_layers - 1)}
                binter = {str(l + 1): self.biases[str(l + 1)] - v_b[l] for l in range(num_layers - 1)}
                
                Y, H, A = self.forwardPropagate(X_train[:, i].reshape(self.img_flat_size, 1), winter, binter)
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[:, i].reshape(self.num_classes, 1))
                
                for l in range(num_layers - 1):
                    deltaw[l] += grad_weights[num_layers - 2 - l]
                    deltab[l] += grad_biases[num_layers - 2 - l]
                
                loss_fn = meanSquaredErrorLoss if self.loss_function == MEAN_SQUARE_KEY else crossEntropyLoss
                LOSS.append(loss_fn(Y_train[:, i].reshape(self.num_classes, 1), Y) + self.L2RegLoss(weight_decay))
                
                num_points_seen += 1
                if num_points_seen % batch_size == 0:
                    v_w = [GAMMA * prev_v_w[l] + learning_rate * deltaw[l] / batch_size for l in range(num_layers - 1)]
                    v_b = [GAMMA * prev_v_b[l] + learning_rate * deltab[l] / batch_size for l in range(num_layers - 1)]
                    
                    for l in range(num_layers - 1):
                        self.weights[str(l + 1)] -= v_w[l]
                        self.biases[str(l + 1)] -= v_b[l]
                    
                    prev_v_w, prev_v_b = v_w, v_b
                    deltaw, deltab = self.initWeight(), self.initBias()
            
            Y_pred = self.predict(self.X_tr, self.N_tr)
            train_loss.append(np.mean(LOSS))
            train_acc.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            
            Y_val_pred = self.predict(self.XVal, self.NVal)
            val_accu.append(accuracy(self.Y_val, Y_val_pred, self.NVal)[0])
            
            loss_fn = meanSquaredErrorLoss if self.loss_function == MEAN_SQUARE_KEY else crossEntropyLoss
            val_loss = np.mean(loss_fn(self.Y_val.T, Y_val_pred.T) + self.L2RegLoss(weight_decay))
            
            elapsed = time.time() - start_time
            printAccuracy(epoch, train_loss[epoch], train_acc[epoch], val_accu[epoch], elapsed, self.alpha)
            epoch += 1

        return train_loss, train_acc, val_accu, Y_pred

    def rmsOptimizer(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        """
        Train the neural network using the RMSProp optimization algorithm.
        """
        X_train, Y_train = self.X_tr[:, :length_dataset], self.Y_train[:, :length_dataset]
        train_loss, train_acc, val_accu = [], [], []
        num_layers, EPS, BETA = len(self.network), 1e-8, 0.9
        v_w, v_b = self.initWeight(), self.initBias()
        num_points_seen, epoch = 0, 0

        while epoch < epochs:
            start_time = time.time()
            offset = np.random.permutation(length_dataset)
            X_train, Y_train = X_train[:, offset], Y_train[:, offset]
            LOSS, deltaw, deltab = [], self.initWeight(), self.initBias()

            for i in range(length_dataset):
                Y, H, A = self.forwardPropagate(X_train[:, i].reshape(self.img_flat_size, 1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[:, i].reshape(self.num_classes, 1))
                deltaw = [grad_weights[num_layers - 2 - j] + deltaw[j] for j in range(num_layers - 1)]
                deltab = [grad_biases[num_layers - 2 - j] + deltab[j] for j in range(num_layers - 1)]
                
                loss_value = (meanSquaredErrorLoss(Y_train[:, i].reshape(self.num_classes, 1), Y) if self.loss_function == MEAN_SQUARE_KEY
                              else crossEntropyLoss(Y_train[:, i].reshape(self.num_classes, 1), Y))
                LOSS.append(loss_value + self.L2RegLoss(weight_decay))
                num_points_seen += 1
                
                if num_points_seen % batch_size == 0:
                    v_w = [BETA * v_w[j] + (1 - BETA) * deltaw[j] ** 2 for j in range(num_layers - 1)]
                    v_b = [BETA * v_b[j] + (1 - BETA) * deltab[j] ** 2 for j in range(num_layers - 1)]
                    
                    self.weights = {str(j + 1): self.weights[str(j + 1)] - (learning_rate / np.sqrt(v_w[j] + EPS)) * deltaw[j]
                                    for j in range(len(self.weights))}
                    self.biases = {str(j + 1): self.biases[str(j + 1)] - (learning_rate / np.sqrt(v_b[j] + EPS)) * deltab[j]
                                   for j in range(len(self.biases))}
                    deltaw, deltab = self.initWeight(), self.initBias()
            
            elapsed = time.time() - start_time
            Y_pred, Y_val_pred = self.predict(self.X_tr, self.N_tr), self.predict(self.XVal, self.NVal)
            train_loss.append(np.mean(LOSS))
            train_acc.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            val_accu.append(accuracy(self.Y_val, Y_val_pred, self.NVal)[0])
            val_loss = np.mean((meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T) if self.loss_function == MEAN_SQUARE_KEY
                                else crossEntropyLoss(self.Y_val.T, Y_val_pred.T)) + self.L2RegLoss(weight_decay))
            printAccuracy(epoch, train_loss[epoch], train_acc[epoch], val_accu[epoch], elapsed, self.alpha)
            epoch += 1
        
        return train_loss, train_acc, val_accu, Y_pred

    def adamOptimizer(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        """
        Train the neural network using the Adam optimization algorithm.
        """
        X_train, Y_train = self.X_tr[:, :length_dataset], self.Y_train[:, :length_dataset]
        train_loss, train_acc, val_accu = [], [], []
        num_layers = len(self.network)
        EPS, BETA1, BETA2 = 1e-8, 0.9, 0.99
        
        # Initialize moment estimates
        m_w, m_b = self.initWeight(), self.initBias()
        v_w, v_b = self.initWeight(), self.initBias()
        
        num_points_seen, epoch = 0, 0
        while epoch < epochs:
            start_time = time.time()
            
            # Shuffle dataset
            indices = np.random.permutation(length_dataset)
            X_train, Y_train = X_train[:, indices], Y_train[:, indices]
            
            loss_per_epoch = []
            deltaw, deltab = self.initWeight(), self.initBias()
            
            for i in range(length_dataset):
                Y, H, A = self.forwardPropagate(X_train[:, i].reshape(self.img_flat_size, 1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[:, i].reshape(self.num_classes, 1))
                
                for j in range(num_layers - 1):
                    deltaw[j] += grad_weights[num_layers - 2 - j]
                    deltab[j] += grad_biases[num_layers - 2 - j]
                
                # Compute loss
                loss = self.L2RegLoss(weight_decay) + (
                    meanSquaredErrorLoss(Y_train[:, i].reshape(self.num_classes, 1), Y)
                    if self.loss_function == MEAN_SQUARE_KEY else
                    crossEntropyLoss(Y_train[:, i].reshape(self.num_classes, 1), Y)
                )
                loss_per_epoch.append(loss)
                
                num_points_seen += 1
                
                if num_points_seen % batch_size == 0:
                    # Update moment estimates
                    for l in range(num_layers - 1):
                        m_w[l] = BETA1 * m_w[l] + (1 - BETA1) * deltaw[l]
                        m_b[l] = BETA1 * m_b[l] + (1 - BETA1) * deltab[l]
                        v_w[l] = BETA2 * v_w[l] + (1 - BETA2) * (deltaw[l] ** 2)
                        v_b[l] = BETA2 * v_b[l] + (1 - BETA2) * (deltab[l] ** 2)
                        
                    # Bias correction
                    m_w_hat = [mw / (1 - BETA1 ** (epoch + 1)) for mw in m_w]
                    m_b_hat = [mb / (1 - BETA1 ** (epoch + 1)) for mb in m_b]
                    v_w_hat = [vw / (1 - BETA2 ** (epoch + 1)) for vw in v_w]
                    v_b_hat = [vb / (1 - BETA2 ** (epoch + 1)) for vb in v_b]
                    
                    # Parameter update
                    for l in range(len(self.weights)):
                        self.weights[str(l + 1)] -= (learning_rate / np.sqrt(v_w_hat[l] + EPS)) * m_w_hat[l]
                        self.biases[str(l + 1)] -= (learning_rate / np.sqrt(v_b_hat[l] + EPS)) * m_b_hat[l]
                    
                    deltaw, deltab = self.initWeight(), self.initBias()
            
            elapsed = time.time() - start_time
            
            # Evaluate accuracy
            Y_pred = self.predict(self.X_tr, self.N_tr)
            train_loss.append(np.mean(loss_per_epoch))
            train_acc.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            
            Y_val_pred = self.predict(self.XVal, self.NVal)
            val_accu.append(accuracy(self.Y_val, Y_val_pred, self.NVal)[0])
            
            val_loss = np.mean(
                meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T) if self.loss_function == MEAN_SQUARE_KEY else
                crossEntropyLoss(self.Y_val.T, Y_val_pred.T)
            ) + self.L2RegLoss(weight_decay)
            
            printAccuracy(epoch, train_loss[epoch], train_acc[epoch], val_accu[epoch], elapsed, self.alpha)
            epoch += 1
        
        return train_loss, train_acc, val_accu, Y_pred

    def nadamOptimizer(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        """
        Train the neural network using the Nadam optimization algorithm.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): Learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - train_loss (list): List of training losses for each epoch.
        - train_acc (list): List of training accuracies for each epoch.
        - val_accu (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.
        """
        X_train, Y_train = self.X_tr[:, :length_dataset], self.Y_train[:, :length_dataset]

        train_loss, train_acc, val_accu = [], [], []
        num_layers = len(self.network)

        # Nadam hyperparameters
        EPS, BETA1, BETA2 = 1e-8, 0.9, 0.99

        # Initialize momentum and velocity terms
        m_w, m_b = self.initWeight(), self.initBias()
        v_w, v_b = self.initWeight(), self.initBias()

        num_points_seen, epoch = 0, 0

        while epoch < epochs:
            start_time = time.time()

            # Shuffle the dataset
            indices = np.random.permutation(length_dataset)
            X_train, Y_train = X_train[:, indices], Y_train[:, indices]

            LOSS, deltaw, deltab = [], self.initWeight(), self.initBias()

            for i in range(length_dataset):
                Y, H, A = self.forwardPropagate(X_train[:, i].reshape(self.img_flat_size, 1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[:, i].reshape(self.num_classes, 1))

                # Accumulate gradients
                deltaw = [grad_weights[j] + deltaw[j] for j in range(num_layers - 1)]
                deltab = [grad_biases[j] + deltab[j] for j in range(num_layers - 1)]

                # Compute loss with L2 regularization
                reg_term = self.L2RegLoss(weight_decay)
                loss_fn = meanSquaredErrorLoss if self.loss_function == MEAN_SQUARE_KEY else crossEntropyLoss
                LOSS.append(loss_fn(Y_train[:, i].reshape(self.num_classes, 1), Y) + reg_term)

                num_points_seen += 1

                # Apply updates at the end of a mini-batch
                if num_points_seen % batch_size == 0:
                    # Update biased first moments (momentum)
                    m_w = [BETA1 * m_w[l] + (1 - BETA1) * deltaw[l] for l in range(num_layers - 1)]
                    m_b = [BETA1 * m_b[l] + (1 - BETA1) * deltab[l] for l in range(num_layers - 1)]

                    # Update biased second moments (velocity)
                    v_w = [BETA2 * v_w[l] + (1 - BETA2) * (deltaw[l] ** 2) for l in range(num_layers - 1)]
                    v_b = [BETA2 * v_b[l] + (1 - BETA2) * (deltab[l] ** 2) for l in range(num_layers - 1)]

                    # Compute bias-corrected estimates
                    m_w_hat = [m_w[l] / (1 - BETA1 ** (epoch + 1)) for l in range(num_layers - 1)]
                    m_b_hat = [m_b[l] / (1 - BETA1 ** (epoch + 1)) for l in range(num_layers - 1)]
                    v_w_hat = [v_w[l] / (1 - BETA2 ** (epoch + 1)) for l in range(num_layers - 1)]
                    v_b_hat = [v_b[l] / (1 - BETA2 ** (epoch + 1)) for l in range(num_layers - 1)]

                    # Compute final parameter updates
                    self.weights = {
                        str(l + 1): self.weights[str(l + 1)] - (learning_rate / (np.sqrt(v_w_hat[l]) + EPS)) *
                        (BETA1 * m_w_hat[l] + (1 - BETA1) * deltaw[l])
                        for l in range(len(self.weights))
                    }
                    self.biases = {
                        str(l + 1): self.biases[str(l + 1)] - (learning_rate / (np.sqrt(v_b_hat[l]) + EPS)) *
                        (BETA1 * m_b_hat[l] + (1 - BETA1) * deltab[l])
                        for l in range(len(self.biases))
                    }

                    # Reset gradients
                    deltaw, deltab = self.initWeight(), self.initBias()

            elapsed = time.time() - start_time

            # Evaluate model performance
            Y_pred = self.predict(self.X_tr, self.N_tr)
            train_loss.append(np.mean(LOSS))
            train_acc.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.XVal, self.NVal)
            val_accu.append(accuracy(self.Y_val, Y_val_pred, self.NVal)[0])

            # Compute validation loss
            reg_term = self.L2RegLoss(weight_decay)
            loss_fn = meanSquaredErrorLoss if self.loss_function == MEAN_SQUARE_KEY else crossEntropyLoss
            val_loss = np.mean(loss_fn(self.Y_val.T, Y_val_pred.T) + reg_term)

            printAccuracy(epoch, train_loss[epoch], train_acc[epoch], val_accu[epoch], elapsed, self.alpha)
            epoch += 1

        return train_loss, train_acc, val_accu, Y_pred

