from mnist import MNIST

import csv
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import sklearn.metrics as metrics
import time

## Global Variables
# Layer sizes
N_IN = 784
N_HID = 200
N_OUT = 10

# Hyperparameters
INITIAL_ALPHA = 2.337e-3
GAMMA = 0.9
SIGMA = 1.45e-3
NUM_EPOCHS = 7
THRESHOLD_EPOCH = 5

VALIDATION_SIZE = 10000

# Saved Model Filenames
V_FILE = "model_v.csv"
W_FILE = "model_w.csv"

## Helper Methods
# Various computational methods
def binarize(X):
    return np.array([np.array([1 if X[i][j] > 0 else 0 for j in range(X.shape[1])]) for i in range(X.shape[0])])

def log_transform(X):
    return np.array([np.array([np.log(X[i][j] + 0.1) for j in range(X.shape[1])]) for i in range(X.shape[0])])

def ReLU(z):
    return np.maximum(z, 0)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def standardize(X):
    return np.array([(X[:,i] - np.mean(X[:,i]))/np.std(X[:,i]) for i in range(X.shape[1])]).T

def apply_relu_grad(x_1):
    x_1[x_1 > 0] = 1
    x_1[x_1 < 0] = 0
    return x_1

def cross_entropy_loss(X, labels, V, W):
    loss = 0
    for i in range(X.shape[0]):
        x_0 = np.reshape(X[i], (1, X[i].shape[0]))
        loss += -np.dot(labels[i], np.transpose(np.log(forward_pass(x_0, V, W)[1][0])))
    return loss

# File I/O Methods
def read_model():
    with open(V_FILE, 'r') as f:
        r1 = csv.reader(f, delimiter=',', quotechar='|')
        V = np.array([np.array([int(x) for x in row]) for row in r1])

    with open(W_FILE, 'r') as g:
        r2 = csv.reader(g, delimiter=',', quotechar='|')
        W = np.array([np.array([int(x) for x in row]) for row in r2])

    return V, W

def write_model(V, W):
    with open(V_FILE, 'w') as f:
        w1 = csv.writer(f, delimiter=',', quotechar='|')
        for i in range(len(V)):
            w1.writerow([V[i][j] for j in range(len(V[i]))])

    with open(W_FILE, 'w') as g:
        w2 = csv.writer(g, delimiter=',', quotechar='|')
        for i in range(len(W)):
            w2.writerow([W[i][j] for j in range(len(W[i]))])

# Data Preprocessing Methods
def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, _ = map(np.array, mndata.load_testing())
    return X_train, labels_train, X_test

def one_hot(labels_train):
    return np.eye(N_OUT)[labels_train]

def normalize(X_train):
    return (X_train - np.mean(X_train)) / 255.0

def shuffle(num_indices):
    indices = np.arange(num_indices)
    np.random.shuffle(indices)

    return indices

def partition_training_data(X_train, labels_train):
    threshold = X_train.shape[0] - VALIDATION_SIZE
    return X_train[:threshold], X_train[threshold:], labels_train[:threshold], labels_train[threshold:]

# Training and Prediction Methods
def train_neural_network(X_train, labels_train):
    V, W = np.random.normal(0.0, SIGMA, (N_HID, N_IN + 1)), np.random.normal(0.0, SIGMA, (N_OUT, N_HID + 1))
    alpha, one_hot_labels = INITIAL_ALPHA, one_hot(labels_train)

    iteration, loss, accuracy = [], [], []
    for i in range(NUM_EPOCHS * X_train.shape[0]):
        if i != 0 and not i % X_train.shape[0]:
            alpha = alpha * GAMMA
        
        # Extract training example
        x_0, y = X_train[i % X_train.shape[0]], one_hot_labels[i % X_train.shape[0]]
        x_0 = np.reshape(x_0, (1, x_0.shape[0]))
        
        # Forward/backwards passes
        x_1, x_2 = forward_pass(x_0, V, W)
        grad_V, grad_W = backprop(x_0, x_1, x_2, y, W)

        # Perform updates
        V, W = V - (alpha * grad_V), W - (alpha * grad_W)

        # Bookkeeping, when necessary
        if not i%10000:
            print(i)
            iteration.append(i)
            loss.append(cross_entropy_loss(X_train, one_hot_labels, V, W))
            accuracy.append(metrics.accuracy_score(labels_train, predict_neural_network(X_train, V, W)))

    return V, W, iteration, loss, accuracy

def forward_pass(x_0, V, W):
    S_1 = np.dot(x_0, np.transpose(V))
    x_1 = np.c_[ReLU(S_1), np.ones((S_1.shape[0], 1))]

    S_2 = np.dot(x_1, np.transpose(W))
    x_2 = softmax(S_2)

    return x_1, x_2

def backprop(x_0, x_1, x_2, y, W):
    grad_W = compute_grad_W(x_1, x_2, y)
    grad_V = compute_grad_V(x_0, x_1, x_2, y, W)
    return grad_V, grad_W

def compute_grad_V(x_0, x_1, x_2, y, W):
    x_1 = apply_relu_grad(x_1)
    return np.dot(np.transpose(np.multiply(np.dot(x_2 - y, W), x_1)), x_0)[:-1]

def compute_grad_W(x_1, x_2, y):
    return np.dot(np.transpose(x_2 - y), x_1)

def predict_neural_network(X_test, V, W):
    return np.argmax(forward_pass(X_test, V, W)[1], axis=1)

# Preprocess
X_train, labels_train, X_test = load_dataset()
shuffled_indices = shuffle(X_train.shape[0])

X_train, labels_train = normalize(X_train[shuffled_indices]), labels_train[shuffled_indices] # Shuffle
X_train, X_validation, labels_train, labels_validation = partition_training_data(X_train, labels_train)

X_train = np.c_[X_train, np.ones((X_train.shape[0], 1))] # Normalize and add bias
X_validation = np.c_[X_validation, np.ones((X_validation.shape[0], 1))]
X_test = np.c_[normalize(X_test), np.ones((X_test.shape[0], 1))]

# Train
start_time = time.time()
V, W, iteration, loss, accuracy = train_neural_network(X_train, labels_train)
end_time = time.time()

total_training_time = end_time - start_time
print("Training completed! Total training time: %d" % total_training_time)

# Predict
prediction_train = predict_neural_network(X_train, V, W)
prediction_validation = predict_neural_network(X_validation, V, W)
prediction_test = predict_neural_network(X_test, V, W)

write_model(V, W)

print("Training accuracy: {0}".format(metrics.accuracy_score(labels_train, prediction_train)))
print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_validation, prediction_validation)))

# Loss Plotting
plt.plot(iteration, loss)
plt.show()

plt.plot(iteration, accuracy)
plt.show()
