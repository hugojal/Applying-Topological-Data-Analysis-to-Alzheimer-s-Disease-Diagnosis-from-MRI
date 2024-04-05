import sklearn.metrics as metrics

pims_h0, pims_h1 = [], []
for i in range(len(X)):
    pd = pds[i]
    pd[:,1:3] = np.clip(pd[:,1:3],a_min=-max_life,a_max=max_life) # clip min/max birth/death
    pim0 = pim.transform(pd[pd[:,0]==0,1:3]) # vectorise PH0 by persistence image
    pim1 = pim.transform(pd[pd[:,0]==1,1:3]) # vectorise PH1 by persistence image
    pims_h0.append(pim0.ravel())
    pims_h1.append(pim1.ravel())

## classification
(trainX, testX, trainY, testY) = train_test_split(pims_h1, Y, test_size = 0.3, random_state = 0, stratify=Y) # use PH1
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(trainX, trainY)

trainPred = clf.predict(trainX)
testPred = clf.predict(testX)
print("Accuracy:", metrics.accuracy_score(testY, testPred))


predY = clf.predict(testX)
print("Confusion matrix\n",confusion_matrix(testY,predY))
print(classification_report(testY,predY))





