import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import argparse
from feedForwardNeuralNetwork import FeedForwardNeuralNetwork
from constant import SIGMOID_KEY,TANH_KEY,RELU_KEY,XAVIER_KEY,RANDOM_KEY,SGD_KEY,MGD_KEY,NAG_KEY,RMSPROP_KEY,ADAM_KEY,NADAM_KEY,CROSS_ENTROPY_KEY,MEAN_SQUARE_KEY
from constant import FASHION_MNIST_DATASET_KEY,MNIST_DATASET_KEY

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--wandb_project", help="Project name for Weights & Biases", default="dl-assigment-1")
parser.add_argument("-we", "--wandb_entity", help="Weights & Biases entity", default="ma23c042")

# Dataset and Model Training Parameters
parser.add_argument("-d", "--dataset", choices=[FASHION_MNIST_DATASET_KEY, MNIST_DATASET_KEY], default=FASHION_MNIST_DATASET_KEY)
parser.add_argument("-e", "--epochs", type=int, choices=[5, 10], default=10)
parser.add_argument("-b", "--batch_size", type=int, choices=[16, 32, 64], default=32)
parser.add_argument("-lr", "--learning_rate", type=float, choices=[1e-3, 1e-4], default=1e-3)

# Loss Function & Optimizer
parser.add_argument("-l", "--loss", choices=[CROSS_ENTROPY_KEY, MEAN_SQUARE_KEY], default=CROSS_ENTROPY_KEY)
parser.add_argument("-o", "--optimizer", choices=[SGD_KEY, MGD_KEY, NAG_KEY, RMSPROP_KEY, ADAM_KEY, NADAM_KEY], default=NADAM_KEY)

# Optimizer Hyperparameters
parser.add_argument("-m", "--momentum", type=float, default=0.5)
parser.add_argument("-beta", "--beta", type=float, default=0.5)
parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
parser.add_argument("-w_d", "--weight_decay", type=float, choices=[0, 0.0005, 0.5], default=0)
parser.add_argument("-w_i", "--weight_init", choices=[RANDOM_KEY, XAVIER_KEY], default=XAVIER_KEY)

# Model Architecture
parser.add_argument("-nhl", "--num_layers", type=int, choices=[3, 4, 5], default=3)
parser.add_argument("-sz", "--hidden_size", type=int, choices=[32, 64, 128], default=128)
parser.add_argument("-a", "--activation", choices=[SIGMOID_KEY, TANH_KEY, RELU_KEY], default=RELU_KEY)

# Parse arguments
args = parser.parse_args()

# Load dataset
(trainIn, trainOut), (testIn, testOut) = (mnist.load_data() if args.dataset == MNIST_DATASET_KEY else fashion_mnist.load_data())

# Dataset Splitting
N_train_full = trainOut.shape[0]
N_train = int(0.9 * N_train_full)
N_validation = int(0.1 * trainOut.shape[0])
N_test = testOut.shape[0]

# Shuffle dataset indices
train_idx = np.random.permutation(N_train_full)
test_idx = np.random.permutation(N_test)

# Train, Validation, and Test Split
trainInFull, trainOutFull = trainIn[train_idx], trainOut[train_idx]
trainIn, trainOut = trainInFull[:N_train], trainOutFull[:N_train]
validIn, validOut = trainInFull[N_train:], trainOutFull[N_train:]
testIn, testOut = testIn[test_idx], testOut[test_idx]

# Best Configurations Dictionary
best_configs = {
    "max_epochs": args.epochs,
    "num_hidden_layers": args.num_layers,
    "num_hidden_neurons": args.hidden_size,
    "weight_decay": args.weight_decay,
    "learning_rate": args.learning_rate,
    "optimizer": args.optimizer,
    "batch_size": args.batch_size,
    "activation": args.activation,
    "initializer": args.weight_init,
    "loss": args.loss
}

# Initialize and Train Model
FFNN = FeedForwardNeuralNetwork(
    num_hidden_layers=best_configs["num_hidden_layers"],
    num_hidden_neurons=best_configs["num_hidden_neurons"],
    X_train_raw=trainInFull,
    Y_train_raw=trainOutFull,
    N_train=N_train_full,
    X_val_raw=validIn,
    Y_val_raw=validOut,
    N_val=N_validation,
    X_test_raw=testIn,
    Y_test_raw=testOut,
    N_test=N_test,
    optimizer=best_configs["optimizer"],
    batch_size=best_configs["batch_size"],
    weight_decay=best_configs["weight_decay"],
    learning_rate=best_configs["learning_rate"],
    max_epochs=best_configs["max_epochs"],
    activation=best_configs["activation"],
    initializer=best_configs["initializer"],
    loss=best_configs["loss"]
)

# Run the optimizer
FFNN.optimizer(FFNN.max_epochs, FFNN.N_tr, FFNN.batch_size, FFNN.alpha)
