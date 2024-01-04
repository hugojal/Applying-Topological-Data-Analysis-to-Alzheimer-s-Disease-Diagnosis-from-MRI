# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier

# Create the sequential model
model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000)

# Fit the model to the training data
model.fit(X_trainval_scaled, Y_trainval)

# Make predictions on the test data
PredictedOutput = model.predict(X_test_scaled)

# Evaluate the model
test_score = model.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, average='macro')
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

# Print the evaluation results
print("Test accuracy:", test_score)
print("Test recall:", test_recall)
print("Test AUC:", test_auc)
