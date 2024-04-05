indices = np.argsort(clf.feature_importances_)
#plt.barh(range(len(indices)), clf.feature_importances_[indices])

idx = [0,1,n,n+1]
nn = len(idx)

fig,axs = plt.subplots(2,nn+1,figsize=(3+nn*3,5))
axs[0,0].imshow(clf.feature_importances_.reshape(pim1.shape),cmap='coolwarm')
axs[0,0].axis('off')
most_important = np.unravel_index(indices[-1], shape=pim1.shape)
life,birth = np.meshgrid(np.linspace(0,max_life,pim1.shape[1]),np.linspace(-max_life,max_life,pim1.shape[0]))
b = birth[most_important]
l = life[most_important]
print("Important feature: birth around ", b,"lifetime around ", l)

# annotate the density of the most contributing feature on the image
# For demonstration, we pick one image from each class.
# the annotation explains how the classifier discerns two classes
for i in range(nn):
  img = X[idx[i]]*255
  pd = pds[idx[i]]
  h = 3*pixel_size
  heat=heat_map(img,pd,h=5,sigma=1,min_life=l-h,max_life=l+h,min_birth =b-h,max_birth = b+h,dimension=1,location='death')
  heat = (heat/max(1,np.max(heat))*255).astype(np.uint8)
  axs[0,i+1].imshow(np.dstack([heat,img//2,img//2]))
  axs[0,i+1].set_title("density")
  axs[0,i+1].axis('off')
  axs[1,i+1].imshow(pim.transform(pd[pd[:,0]==1,1:3]))
  axs[1,i+1].set_title("image of H{}".format(1))
  axs[1,i+1].axis('off')




IMG_DIR = "./images/"  ## dir containing images for a 3D example; all the images must be of the same dimension

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Define the filepath to your NIfTI scan
scanFilePath = 'ADNI/023_S_0042/MPR-R__GradWarp__B1_Correction__N3__Scaled/2005-10-31_12_03_30.0/I31084/ADNI_023_S_0042_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20061130173634563_S8852_I31084.nii'

#Load the scan and extract data using nibabel
scan = nib.load(scanFilePath)
scanArray = scan.get_fdata()

#Get and print the scan's shape
scanArrayShape = scanArray.shape
# print('The scan data array has the shape: ', scanArrayShape)

#Get and print the scan's header
scanHeader = scan.header
# print('The scan header is as follows: \n', scanHeader)

#Display scan array's middle slices
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate proper aspect ratios
pixDim = scanHeader['pixdim'][1:4]
aspectRatios = [pixDim[1]/pixDim[2],pixDim[0]/pixDim[2],pixDim[0]/pixDim[1]]
# print('The required aspect ratios are: ', aspectRatios)

#Display scan array's middle slices with proper aspect ratio
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array w/ Proper Aspect Ratio (Middle Slices)')
# axs[0].imshow(scanArray[scanArrayShape[0]//2,:,:], aspect = aspectRatios[0], cmap='gray')
# axs[1].imshow(scanArray[:,scanArrayShape[1]//2,:], aspect = aspectRatios[1], cmap='gray')
# axs[2].imshow(scanArray[:,:,scanArrayShape[2]//2], aspect = aspectRatios[2], cmap='gray')
# fig.tight_layout()
# plt.show()

#Calculate new image dimensions from aspect ratio
newScanDims = np.multiply(scanArrayShape, pixDim)
newScanDims = (round(newScanDims[0]),round(newScanDims[1]),round(newScanDims[2]))
# print('The new scan size is: ', newScanDims)

#Set the output file path
outputPath = 'output/'

#Iterate and save scan slices along 0th dimension
for i in range(scanArrayShape[0]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[i,:,:], (newScanDims[2],newScanDims[1]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim0_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 1st dimension
for i in range(scanArrayShape[1]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,i,:], (newScanDims[2],newScanDims[0]))
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim1_Slice'+str(i)+'.png', outputArray)

#Iterate and save scan slices along 2nd dimension
for i in range(scanArrayShape[2]):
    #Resample the slice
    outputArray = cv2.resize(scanArray[:,:,i], (newScanDims[1],newScanDims[0]))
    #Rotate slice clockwise 90 degrees
    outputArray = cv2.rotate(outputArray, cv2.ROTATE_90_CLOCKWISE)
    #Save the slice as .png image
    cv2.imwrite(outputPath+'Dim2_Slice'+str(i)+'.png', outputArray)




pd = cripser.computePH(img)
pds = [pd[pd[:,0] == i] for i in range(3)]
print("Betti numbers: ",[len(pds[i]) for i in range(len(pds))])

# T-construction of the original image (pixel value filtration)
pdt = tcripser.computePH(img)
pdst = [pdt[pdt[:,0] == i] for i in range(3)]
print("Betti numbers: ",[len(pdst[i]) for i in range(len(pdst))])

## plot persistent diagram using persim
fig,axs = plt.subplots(1,2)
persim.plot_diagrams([p[:,1:3] for p in pds], ax=axs[0], title='V-construction')
persim.plot_diagrams([p[:,1:3] for p in pdst], ax=axs[1], title='T-construction')




%%timeit -r 3
# compute PH for the T-construction by GUDHI
gd = gudhi.CubicalComplex(top_dimensional_cells=img)
#gd = gudhi.CubicalComplex(vertices=img)
#    gd.compute_persistence()
res = gd.persistence(2,0) # coeff = 2
# print("Betti numbers: ", gd.persistent_betti_numbers(np.inf,-np.inf))





%%timeit -r 3
# compute PH for the T-construction by CubicalRipser
pdt = tcripser.computePH(img, fortran_order=True)
pdst = [pdt[pdt[:,0] == i] for i in range(3)]
# print("Betti numbers: ",[len(pdst[i]) for i in range(len(pdst))])





n=200  # number of samples
t = np.linspace(0,1,100)
a=np.random.uniform(low=1,high=2,size=n)  # amplitude
b=np.random.uniform(low=1.5,high=7,size=n) # period <= our target for regression
c=np.random.uniform(low=-np.pi,high=np.pi,size=n) # phase
# create different sine curves with noise
X = [a[i]*np.sin(2*np.pi*b[i]*t+c[i]) + np.random.normal(scale=0.1,size=len(t)) for i in range(n)]
for i in range(5):
    plt.plot(t,X[i])






import persim,cripser
import PersistenceImages.persistence_images as pimg

#pim=persim.PersImage(pixels=[10,10], spread=1) ## for persim
pim = pimg.PersistenceImager(birth_range=(0,5), pers_range=(0,5), pixel_size=0.5)
#print(pim)
pds, pims = [], []
for i in range(len(X)):
    pd = cripser.computePH(X[i])[:,1:3]    # compute PH
    pds.append(np.clip(pd,a_min=-2,a_max=2))  # clip min/max birth/death
    pims.append(pim.transform(pds[i])) # vectorise by persistence image
for i in range(4):   # plot persistence images for the first four samples
    ax = plt.subplot(240+i+1)
    ax.imshow(pims[i])
    #pim.show(pims[i], ax) ## for persim





from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# train-test split
(trainX, testX, trainY, testY) = train_test_split(np.array(pims).reshape(n,-1), b, test_size = 0.3, random_state = 0) # use PH features

# fit model
clf = linear_model.Lasso(alpha=0.0001,max_iter=10000)
clf.fit(trainX, trainY)

# prediction: the result is reasonably good
trainPred = clf.predict(trainX)
testPred = clf.predict(testX)
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(trainY,trainPred), mean_squared_error(testY,testPred)) )
print('R2 train : %.3f, test : %.3f' % (r2_score(trainY,trainPred), r2_score(testY,testPred)) )

plt.figure(figsize=(12,8))
plt.plot(testY,label="true")
plt.plot(testPred, label="pred")
plt.legend()
plt.show()






plt.figure(figsize = (10, 7))
plt.scatter(trainPred, trainPred - trainY, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'Training data')
plt.scatter(testPred, testPred - testY, c = 'lightgreen', marker = 's', s = 35, alpha = 0.7, label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 1, xmax = 8, lw = 2, color = 'red')
plt.xlim([1, 8])
plt.show()







(trainX, testX, trainY, testY) = train_test_split(X, b, test_size = 0.3, random_state = 0)  # see what happens if we use the input data directly

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




