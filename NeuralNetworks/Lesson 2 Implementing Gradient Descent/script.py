import numpy as np
import pandas as pd

admissions = pd.read_csv('C:/Users/User/Desktop/DeepLearningNanodegree/NeuralNetworks/Lesson 2 Implementing Gradient Descent/binary.csv')
# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

data

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.loc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
weights

# Neural Network hyperparameters
epochs = 10
learnrate = 0.5

features

targets
output = sigmoid(np.dot(features.values, weights))
error = targets - output
error

error * output * (1 - output)



for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        
        output = sigmoid(np.dot(x, weights))
        error = y - output
        error_term = error * output * (1 - output)
        del_w += error_term * x

    weights += learnrate * del_w / n_records

    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))