import gudhi.point_cloud.timedelay
embedder = gudhi.point_cloud.timedelay.TimeDelayEmbedding(dim=2, delay=3, skip=1)
X_takens = embedder.transform(X)

pim = pimg.PersistenceImager(birth_range=(0,2), pers_range=(0,2), pixel_size=0.2)
pds, pims = [], []
for i in range(len(X)):
    pd = ripser(X_takens[i])['dgms']    # compute PH
    pim0 = pim.transform(np.clip(pd[0],a_min=0,a_max=2)) #PH_0
    pim1 = pim.transform(np.clip(pd[1],a_min=0,a_max=2)) #PH_1
    pims.append(np.concatenate([pim0,pim1])) # combine PH_0 and PH_1 to make a feature vector
for i in range(4):   # plot persistence images
    ax = plt.subplot(240+i+1)
    ax.imshow(pims[i])
    #pim.show(pims[i], ax) ## for persim

## linear regression fot the period
# train-test split
(trainX, testX, trainY, testY) = train_test_split(np.array(pims).reshape(n,-1), b, test_size = 0.3, random_state = 0) # use PH features

# fit model
clf = linear_model.Lasso(alpha=0.0001,max_iter=10000)
clf.fit(trainX, trainY)


trainPred = clf.predict(trainX)
testPred = clf.predict(testX)
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(trainY,trainPred), mean_squared_error(testY,testPred)) )
print('R2 train : %.3f, test : %.3f' % (r2_score(trainY,trainPred), r2_score(testY,testPred)) )

plt.figure(figsize=(12,8))
plt.plot(testY,label="true")
plt.plot(testPred, label="pred")
plt.legend()
plt.show()
