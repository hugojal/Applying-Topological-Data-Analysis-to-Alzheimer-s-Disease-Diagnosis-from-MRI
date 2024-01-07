!pip install scikit-learn

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, roc_curve, auc

best_score = 0

M = 14
lr = 0.001

for Q in range(2, 15, 2):  # combines M trees
    for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
        boostModel = GradientBoostingClassifier(n_estimators=Q, learning_rate=learning_rate, random_state=0)

        # perform cross-validation
        scores = cross_val_score(boostModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')

        # compute mean cross-validation accuracy
        score = np.mean(scores)

        if score > best_score:
            best_score = score

# select the best model based on cross-validation performance
SelectedBoostModel = GradientBoostingClassifier(n_estimators=M, learning_rate=lr, random_state=0).fit(X_trainval_scaled, Y_trainval)

# make predictions on the test set
PredictedOutput = SelectedBoostModel.predict(X_test_scaled)

# evaluate the model on the test set
test_score = SelectedBoostModel.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, average='macro')
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

# print the evaluation results
print("val accuracy:", best_score)
print("Test accuracy", test_score)

m = 'GradientBoosting'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])
