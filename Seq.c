import keras.models
import keras
import keras.layers
from keras.layers import Dense

best_score = 0

for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
    model = keras.models.Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_trainval_scaled.shape[1],)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # perform cross-validation
    scores = model.evaluate(X_trainval_scaled, Y_trainval)

    # compute mean cross-validation accuracy
    score = np.mean(scores)

    if score > best_score:
        best_score = score

SelectedModel = keras.models.Sequential()
SelectedModel.add(Dense(10, activation='relu', input_shape=(X_trainval_scaled.shape[1],)))
SelectedModel.add(Dense(10, activation='relu'))
SelectedModel.add(Dense(1, activation='sigmoid'))

SelectedModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

SelectedModel.fit(X_trainval_scaled, Y_trainval, epochs=100, batch_size=32)
PredictedOutput = SelectedModel.predict(X_test_scaled, batch_size=32)
test_score, test_recall, test_auc = SelectedModel.evaluate(X_test_scaled, Y_test)[1], recall_score(Y_test, PredictedOutput.reshape(-1,), average='macro'), auc(roc_curve(Y_test, PredictedOutput, pos_label=1)[0], roc_curve(Y_test, PredictedOutput)[1])
m = 'Deep Learning Model'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])
