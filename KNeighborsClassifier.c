# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# Create the KNeighborsClassifier model
best_score = 0
n_neighbors = 0
kfolds = 5
for n in range(1, 21):
    knnModel = KNeighborsClassifier(n_neighbors=n)

    # perform cross-validation
    scores = cross_val_score(knnModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')

    # compute mean cross-validation accuracy
    score = np.mean(scores)

    if score > best_score:
        best_score = score
        n_neighbors_best = n

# Fit the model to the training data
SelectedKNNModel = KNeighborsClassifier(n_neighbors=n_neighbors_best).fit(X_trainval_scaled, Y_trainval)

# Make predictions on the test data
PredictedOutput = SelectedKNNModel.predict(X_test_scaled)

# Evaluate the model
test_score = SelectedKNNModel.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, average='macro')
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

# Print the evaluation results
print("Test accuracy:", test_score)
print("Test recall:", test_recall)
print("Test AUC:", test_auc)
