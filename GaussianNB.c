# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc



# Create the GaussianNB model
best_score = 0
var_smoothing = 0
kfolds = 5
for var_smoothing in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
    gnbModel = GaussianNB(var_smoothing=var_smoothing)

    # perform cross-validation
    scores = cross_val_score(gnbModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')

    # compute mean cross-validation accuracy
    score = np.mean(scores)

    if score > best_score:
        best_score = score
        var_smoothing_best = var_smoothing

# Fit the model to the training data
SelectedGaussianNBModel = GaussianNB(var_smoothing=var_smoothing_best).fit(X_trainval_scaled, Y_trainval)

# Make predictions on the test data
PredictedOutput = SelectedGaussianNBModel.predict(X_test_scaled)

test_score = SelectedGaussianNBModel.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, average='macro')
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

# print the evaluation results
print("val accuracy:", best_score)
print("Test accuracy", test_score)

m = 'GaussianNB'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])
