import sys

import numpy as np

#filename = sys.argv[1]
filename = "data_singlevar.txt"
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)
y_train_pred = linear_regressor.predict(X_train)

# Predict the output
y_test_pred = linear_regressor.predict(X_test)

# Plot outputs
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, y_train_pred, color='black', linewidth=2)
plt.plot(X_test, y_test_pred, color='blue', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()

# Measure performance
import sklearn.metrics as sm

print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) )
print ("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2) )
print ("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2) )
print ("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2) )
print ("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2) )

# Model persistence
import pickle #use pickle in python3 instead of cPickle in python2

output_model_file = '1_6_model_linear_regr.pkl'

with open(output_model_file, 'wb') as f: # need to be wb mode to write the byte mode instead of string mode
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print ("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2) )

