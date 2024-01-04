# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# Create the SVC model
best_score = 0
C = 0
gamma = 0
kfolds = 5
for C in [0.001, 0.01, 0.1, 1, 10]:
    for gamma in [0.001, 0.01, 0.1, 1, 10]:
        svmModel = SVC(C=C, gamma=gamma, random_state=0)

        # perform cross-validation
        scores = cross_val_score(svmModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')

        # compute mean cross-validation accuracy
        score = np.mean(scores)

        if score > best_score:
            best_score = score
            C_best = C
            gamma_best = gamma

# Fit the model to the training data
SelectedSVMModel = SVC(C=C_best, gamma=gamma_best, random_state=0).fit(X_trainval_scaled, Y_trainval)

# Make predictions on the test data
PredictedOutput = SelectedSVMModel.predict(X_test_scaled)

test_score = SelectedSVMModel.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, average='macro')
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

# print the evaluation results
print("val accuracy:", best_score)
print("Test accuracy", test_score)

m = 'SVM'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])
