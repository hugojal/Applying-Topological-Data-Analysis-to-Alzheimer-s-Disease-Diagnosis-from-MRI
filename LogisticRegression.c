import sklearn.linear_model
from sklearn.linear_model import LogisticRegression

best_score = 0

for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
    boostModel = LogisticRegression(C=learning_rate, random_state=0)

    # perform cross-validation
    scores = cross_val_score(boostModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')

    # compute mean cross-validation accuracy
    score = np.mean(scores)

    if score > best_score:
        best_score = score

SelectedBoostModel = LogisticRegression(C=lr, random_state=0).fit(X_trainval_scaled, Y_trainval )

PredictedOutput = SelectedBoostModel.predict(X_test_scaled)
test_score = SelectedBoostModel.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, average='macro')
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)
print("val accuracy:", best_score*100)
print("Test accuracy", test_score*100)


m = 'LogisticRegression'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])
